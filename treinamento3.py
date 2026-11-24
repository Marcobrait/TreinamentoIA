"""Versao do treinamento com suporte adicional a classificacao de imagem.

Mantem ``web_treinamento2 - Copia.py`` intacto. Execute este arquivo para usar
os tres tipos de projeto: deteccao, segmentacao e classificacao.
"""
from __future__ import annotations

import importlib.util
import re
import shutil
import sys
import threading
from datetime import datetime
from pathlib import Path


BASE_FILE = Path(__file__).with_name('web_treinamento2 - Copia.py')
spec = importlib.util.spec_from_file_location('treinamento_base', BASE_FILE)
if spec is None or spec.loader is None:
    raise RuntimeError(f'Nao foi possivel carregar {BASE_FILE.name}.')
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


def _task(pp):
    if pp.task_file.exists():
        value = pp.task_file.read_text(encoding='utf-8').strip()
        if value in {'detect', 'segment', 'classify'}:
            return value
    return 'detect'


def _create_project(nome, classes_iniciais=None, task='detect'):
    if task not in {'detect', 'segment', 'classify'}:
        raise ValueError('Tipo de treinamento invalido.')
    pp = base.ProjectPaths(base.slugify_projeto(nome))
    if not pp.nome:
        raise ValueError('Nome do modelo nao pode ser vazio.')
    if pp.root.exists():
        raise ValueError('Ja existe um modelo com esse nome.')
    pp.ensure()
    base.save_project_task(pp, task)
    classes = [str(c).strip() for c in (classes_iniciais or []) if str(c).strip()]
    if classes:
        base.save_classes(pp, classes)
    return pp


def _class_dir(pp, split, class_index):
    classes = base.load_classes(pp)
    if not 0 <= class_index < len(classes):
        raise ValueError('Classe invalida.')
    # O prefixo numerico torna os nomes estaveis mesmo se houver espacos/acento.
    safe = re.sub(r'[^\w.-]+', '_', classes[class_index], flags=re.UNICODE).strip('._') or 'classe'
    return pp.dataset / 'classify' / split / f'{class_index:03d}_{safe}'


def _resolve_base_model(pp, modelo_base):
    if modelo_base == '__ultimo__':
        candidates = sorted(
            (d for d in pp.runs.iterdir() if d.is_dir() and (d / 'weights' / 'last.pt').exists()),
            key=lambda d: d.stat().st_mtime, reverse=True,
        ) if pp.runs.exists() else []
        if not candidates:
            raise ValueError('Nenhum treinamento anterior encontrado para continuar neste modelo.')
        return str(candidates[0] / 'weights' / 'last.pt')
    if _task(pp) != 'classify':
        return _original_resolve_base_model(pp, modelo_base)
    allowed = {'yolov8n-cls.pt', 'yolov8s-cls.pt', 'yolov8m-cls.pt'}
    if modelo_base not in allowed:
        raise ValueError('Modelo base de classificacao invalido.')
    return modelo_base


def _run_classification_training(pp, params):
    run_name = datetime.now().strftime('treino_%Y%m%d_%H%M%S')
    run_dir = pp.runs / run_name
    try:
        classes = base.load_classes(pp)
        if len(classes) < 2:
            raise ValueError('Classificacao exige pelo menos duas classes.')
        root = pp.dataset / 'classify'
        train_count = sum(1 for p in (root / 'train').rglob('*') if p.is_file()) if (root / 'train').exists() else 0
        val_count = sum(1 for p in (root / 'val').rglob('*') if p.is_file()) if (root / 'val').exists() else 0
        if not train_count:
            raise ValueError('Nao ha imagens classificadas em "Treino".')
        if not val_count:
            raise ValueError('Nao ha imagens classificadas em "Validacao". Classifique ao menos uma imagem para validar.')

        model_path = _resolve_base_model(pp, params['modelo_base'])
        base.training_state.add_log(f'Carregando modelo de classificacao: {model_path}')
        model = base.YOLO(model_path)

        def on_start(trainer):
            device = base.descrever_dispositivo(trainer)
            base.training_state.set_device(device)
            base.training_state.add_log(f"Classificacao iniciada: {params['epocas']} epocas, imgsz={params['imgsz']}, batch={params['batch']}, workers={params['workers']}, dispositivo={device}.")

        def on_epoch_start(_trainer):
            base.training_state.mark_epoch_start()

        def on_epoch_end(trainer):
            elapsed = base.training_state.mark_epoch_end()
            epoch = int(getattr(trainer, 'epoch', 0)) + 1
            total = int(getattr(trainer, 'epochs', params['epocas']))
            metrics = base._extract_metrics(trainer)
            base.training_state.update_epoch(epoch, total, metrics)
            suffix = f' ({base.formatar_duracao(elapsed)})' if elapsed is not None else ''
            base.training_state.add_log(f'Epoca {epoch}/{total}{suffix}')

        model.add_callback('on_train_start', on_start)
        model.add_callback('on_train_epoch_start', on_epoch_start)
        model.add_callback('on_fit_epoch_end', on_epoch_end)
        model.train(data=str(root.resolve()), epochs=int(params['epocas']), imgsz=int(params['imgsz']),
                    batch=params['batch'], workers=int(params['workers']), project=str(pp.runs.resolve()),
                    name=run_name, exist_ok=True, verbose=False)
        base.training_state.add_log('Treinamento de classificacao concluido.')
        base.training_state.finish(run_dir)
    except Exception as exc:
        if (run_dir / 'weights' / 'best.pt').exists() or (run_dir / 'weights' / 'last.pt').exists():
            base.training_state.add_log(f'Aviso apos salvar pesos: {exc}')
            base.training_state.finish(run_dir)
        else:
            base.training_state.add_log(f'ERRO: {exc}')
            base.training_state.fail(exc)


def _run_training(pp, params):
    if _task(pp) == 'classify':
        return _run_classification_training(pp, params)
    return _original_run_training(pp, params)


_original_resolve_base_model = base._resolve_base_model
_original_run_training = base._run_training
base.load_project_task = _task
base.create_project = _create_project
base._resolve_base_model = _resolve_base_model
base._run_training = _run_training


CLASSIFY_UI = r'''
<script>
(() => {
  const radioHost = document.querySelector('.radio-row');
  if (radioHost) radioHost.insertAdjacentHTML('beforeend', '<label><input type="radio" name="novo-projeto-tipo" value="classify" /> Classificacao (uma nota por imagem)</label>');
  const baseSelector = window.atualizarSeletorModeloBase;
  window.atualizarSeletorModeloBase = function () {
    if (projetoTask !== 'classify') return baseSelector();
    const sel = document.getElementById('cfg-modelo');
    const current = sel.value;
    sel.innerHTML = [['yolov8n-cls.pt','YOLOv8n-cls (rapido, leve, precisa internet)'],['yolov8s-cls.pt','YOLOv8s-cls (equilibrado, precisa internet)'],['yolov8m-cls.pt','YOLOv8m-cls (mais preciso, precisa internet)']].map(x => `<option value="${x[0]}">${x[1]}</option>`).join('') + '<option value="__ultimo__">Continuar do ultimo treinamento deste modelo</option>';
    if ([...sel.options].some(o => o.value === current)) sel.value = current;
  };
  const originalOpen = window.abrirAnotacao;
  window.abrirAnotacao = async function(pasta, nome) {
    if (projetoTask !== 'classify') return originalOpen(pasta, nome);
    const classes = state.classes || [];
    if (!classes.length) return alert('Cadastre uma classe antes de classificar a imagem.');
    const choice = prompt(`Classe da imagem "${nome}":\n` + classes.map((c,i) => `${i + 1} - ${c}`).join('\n'));
    if (choice === null) return;
    const idx = Number(choice) - 1;
    if (!Number.isInteger(idx) || idx < 0 || idx >= classes.length) return alert('Escolha uma classe valida pelo numero.');
    const split = confirm('Enviar para VALIDACAO?\nOK = Validacao | Cancelar = Treino') ? 'validacao' : 'treino';
    const res = await fetch('/classify/anotar', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({projeto:projetoAtual, pasta, nome, classe:idx, destino:split})});
    const data = await res.json();
    if (!data.ok) return alert(data.erro || 'Falha ao classificar imagem.');
    alert(`Imagem classificada como ${data.classe} em ${split}.`);
    if (typeof carregarImagens === 'function') carregarImagens();
  };
  document.addEventListener('click', (ev) => {
    if (projetoTask !== 'classify') return;
    const target = ev.target.closest('#btn-salvar-anotacao, #btn-auto-detectar');
    if (target) { ev.preventDefault(); ev.stopImmediatePropagation(); alert('Em Classificacao, clique numa miniatura e escolha uma unica classe.'); }
  }, true);
})();
</script>
'''
base.PAGE_HTML = base.PAGE_HTML.replace('</body>', CLASSIFY_UI + '</body>')


def create_app(camera_manager, titulo):
    app = base.create_app(camera_manager, titulo)

    @app.post('/classify/anotar')
    def classify_image():
        data = base.request.get_json(silent=True) or {}
        pp = base.get_project(data.get('projeto', ''))
        if _task(pp) != 'classify':
            return base.jsonify({'ok': False, 'erro': 'Este projeto nao e de classificacao.'}), 400
        try:
            index = int(data.get('classe'))
            destination = str(data.get('destino', 'treino'))
            if destination not in {'treino', 'validacao'}:
                raise ValueError('Destino invalido.')
            target_dir = _class_dir(pp, 'train' if destination == 'treino' else 'val', index)
        except (TypeError, ValueError) as exc:
            return base.jsonify({'ok': False, 'erro': str(exc) or 'Classe invalida.'}), 400
        source_folder = {'pendentes': pp.capturas, 'treino': pp.images_train, 'validacao': pp.images_val}.get(str(data.get('pasta', 'pendentes')))
        source = source_folder / base.safe_name(data.get('nome', '')) if source_folder else None
        if source is None or not source.exists() or not source.is_file():
            return base.jsonify({'ok': False, 'erro': 'Imagem nao encontrada.'}), 404
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        if target.exists():
            target = target_dir / f'{source.stem}_{datetime.now().strftime("%H%M%S_%f")}{source.suffix}'
        shutil.move(str(source), str(target))
        return base.jsonify({'ok': True, 'classe': base.load_classes(pp)[index]})

    return app


def main():
    parser = base.load_config(base.CONFIG_FILE)
    camera_manager = base.CameraManager(parser)
    try:
        camera_manager.iniciar()
    except RuntimeError as exc:
        print(f'[AVISO] {exc}')
    host = parser.get('web', 'host', fallback='0.0.0.0')
    port = parser.getint('web', 'porta', fallback=5011)
    title = parser.get('web', 'titulo', fallback='Treinamento de Modelo') + ' - Classificacao'
    app = create_app(camera_manager, title)
    try:
        print(f'Web de treinamento rodando em http://{host}:{port}')
        app.run(host=host, port=port, debug=False, threaded=True)
    finally:
        camera_manager.stop()


if __name__ == '__main__':
    main()
