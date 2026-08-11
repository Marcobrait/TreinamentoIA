# python.exe -m venv .venv
# .\.venv\Scripts\Activate.ps1
# python -m pip install --upgrade pip
# python -m pip install -r requirements.txt

import configparser
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import cv2
import torch
import yaml
from flask import Flask, Response, abort, jsonify, render_template_string, request, send_file
from ultralytics import YOLO


CONFIG_FILE = Path('config.cfg')
PROJETOS_DIR = Path('projetos')
DEFAULT_PROJETO = 'BTS'
LOCAL_BASE_MODEL = Path('yolov8n.pt')
IMAGE_EXTENSIONS = ('*.jpg', '*.jpeg', '*.png')

PALETTE = [
    '#22c55e', '#f59e0b', '#3b82f6', '#ef4444',
    '#a855f7', '#14b8a6', '#eab308', '#ec4899',
]

WINDOWS_RESERVED = {
    'CON', 'PRN', 'AUX', 'NUL',
    'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
    'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9',
}


def slugify_projeto(nome: str) -> str:
    nome = (nome or '').strip()
    for ch in '<>:"/\\|?*':
        nome = nome.replace(ch, '_')
    nome = ' '.join(nome.split())
    nome = nome.strip(' .')
    if nome.upper() in WINDOWS_RESERVED:
        nome = f'_{nome}'
    return nome[:80]


class ProjectPaths:
    def __init__(self, nome: str) -> None:
        self.nome = nome
        self.root = PROJETOS_DIR / nome
        self.dataset = self.root / 'dataset'
        self.capturas = self.dataset / 'capturas'
        self.images_train = self.dataset / 'images' / 'train'
        self.images_val = self.dataset / 'images' / 'val'
        self.labels_train = self.dataset / 'labels' / 'train'
        self.labels_val = self.dataset / 'labels' / 'val'
        self.classes_file = self.dataset / 'classes.txt'
        self.runs = self.root / 'runs_treinamento'

    def ensure(self) -> None:
        for d in (self.capturas, self.images_train, self.images_val, self.labels_train, self.labels_val, self.runs):
            d.mkdir(parents=True, exist_ok=True)
        if not self.classes_file.exists():
            self.classes_file.write_text('', encoding='utf-8')

    def exists(self) -> bool:
        return self.root.exists()


def list_projetos() -> list:
    if not PROJETOS_DIR.exists():
        return []
    return sorted(d.name for d in PROJETOS_DIR.iterdir() if d.is_dir())


def get_project(nome: str) -> ProjectPaths:
    slug = slugify_projeto(nome) or DEFAULT_PROJETO
    pp = ProjectPaths(slug)
    pp.ensure()
    return pp


def create_project(nome: str, classes_iniciais=None) -> ProjectPaths:
    slug = slugify_projeto(nome)
    if not slug:
        raise ValueError('Nome do modelo nao pode ser vazio.')
    pp = ProjectPaths(slug)
    if pp.root.exists():
        raise ValueError('Ja existe um modelo com esse nome.')
    pp.ensure()
    classes_iniciais = [c.strip() for c in (classes_iniciais or []) if c.strip()]
    if classes_iniciais:
        save_classes(pp, classes_iniciais)
    return pp


def _migrate_legacy_layout() -> None:
    """Compatibilidade: versoes antigas guardavam tudo em dataset/ e
    runs_treinamento/ na raiz, sem suporte a varios modelos. Se ainda
    existir esse layout antigo e a pasta projetos/ nao existir, move tudo
    para projetos/BTS automaticamente, sem perder nada que ja foi capturado."""
    legacy_dataset = Path('dataset')
    legacy_runs = Path('runs_treinamento')
    if PROJETOS_DIR.exists():
        return
    PROJETOS_DIR.mkdir(parents=True, exist_ok=True)
    if not legacy_dataset.exists():
        return
    destino = ProjectPaths(DEFAULT_PROJETO)
    destino.root.mkdir(parents=True, exist_ok=True)
    shutil.move(str(legacy_dataset), str(destino.dataset))
    if legacy_runs.exists():
        shutil.move(str(legacy_runs), str(destino.runs))
    destino.ensure()


_migrate_legacy_layout()


def load_config(config_path: Path) -> configparser.ConfigParser:
    if not config_path.exists():
        raise FileNotFoundError(f'Arquivo de configuracao nao encontrado: {config_path.resolve()}')
    parser = configparser.ConfigParser()
    parser.read(config_path, encoding='utf-8-sig')
    if 'camera' not in parser:
        raise ValueError('Secao [camera] nao encontrada no config.cfg')
    return parser


def load_rtsp_url(parser: configparser.ConfigParser) -> str:
    cam = parser['camera']
    ip = cam.get('ip', '').strip()
    usuario = cam.get('usuario', '').strip()
    senha = cam.get('senha', '').strip()
    if not ip or not usuario or not senha:
        raise ValueError('Preencha ip, usuario e senha na secao [camera]')
    porta = cam.get('porta', '554').strip() or '554'
    caminho = cam.get('caminho', '/cam/realmonitor?channel=1&subtype=0').strip()
    if not caminho.startswith('/'):
        caminho = '/' + caminho
    usuario_escapado = quote(usuario, safe='%')
    senha_escapada = quote(senha, safe='%')
    return f'rtsp://{usuario_escapado}:{senha_escapada}@{ip}:{porta}{caminho}'


class CameraStream:
    def __init__(self, rtsp_url: str) -> None:
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(rtsp_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock = threading.Lock()
        self._frame = None
        self._running = False
        self._thread = None
        self._fail_count = 0

    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            try:
                ok, frame = self.cap.read()
                if not ok:
                    self._fail_count += 1
                    if self._fail_count >= 100:
                        self.cap.release()
                        self.cap = cv2.VideoCapture(self.rtsp_url)
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        self._fail_count = 0
                    time.sleep(0.01)
                    continue
                self._fail_count = 0
                with self._lock:
                    self._frame = frame
            except Exception:
                time.sleep(0.5)

    def get_frame(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()


def load_classes(pp: ProjectPaths) -> list:
    if not pp.classes_file.exists():
        return []
    return [line.strip() for line in pp.classes_file.read_text(encoding='utf-8').splitlines() if line.strip()]


def save_classes(pp: ProjectPaths, classes: list) -> None:
    text = '\n'.join(classes)
    pp.classes_file.write_text(text + ('\n' if text else ''), encoding='utf-8')


def safe_name(raw: str) -> str:
    return Path(str(raw)).name


def detectar_ambiente() -> dict:
    info = {'gpu_disponivel': False, 'gpu_nome': None, 'gpu_memoria_gb': None, 'cuda_versao': None}
    try:
        if torch.cuda.is_available():
            info['gpu_disponivel'] = True
            info['gpu_nome'] = torch.cuda.get_device_name(0)
            info['gpu_memoria_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)
            info['cuda_versao'] = torch.version.cuda
    except Exception:
        pass
    return info


def descrever_dispositivo(trainer) -> str:
    device_str = str(getattr(trainer, 'device', '')).lower()
    if 'cuda' not in device_str:
        return 'CPU'
    try:
        idx = trainer.device.index if getattr(trainer.device, 'index', None) is not None else 0
        return f'GPU ({torch.cuda.get_device_name(idx)})'
    except Exception:
        return f'GPU ({device_str})'


class TrainingState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.status = 'ocioso'
        self.projeto = None
        self.epoch = 0
        self.epochs_total = 0
        self.metrics = {}
        self.log = []
        self.run_dir = None
        self.error = None
        self.started_at = None
        self.finished_at = None
        self.device = None

    def reset_for_start(self, projeto: str, epochs_total: int) -> None:
        with self.lock:
            self.running = True
            self.status = 'rodando'
            self.projeto = projeto
            self.epoch = 0
            self.epochs_total = epochs_total
            self.metrics = {}
            self.log = []
            self.run_dir = None
            self.error = None
            self.started_at = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
            self.finished_at = None
            self.device = None

    def set_device(self, device: str) -> None:
        with self.lock:
            self.device = device

    def add_log(self, text: str) -> None:
        with self.lock:
            self.log.append(text)
            self.log = self.log[-200:]

    def update_epoch(self, epoch: int, epochs_total: int, metrics: dict) -> None:
        with self.lock:
            self.epoch = epoch
            self.epochs_total = epochs_total
            self.metrics = metrics

    def finish(self, run_dir) -> None:
        with self.lock:
            self.running = False
            self.status = 'concluido'
            self.run_dir = str(run_dir) if run_dir else None
            self.finished_at = datetime.now().strftime('%d/%m/%Y %H:%M:%S')

    def fail(self, error) -> None:
        with self.lock:
            self.running = False
            self.status = 'erro'
            self.error = str(error)
            self.finished_at = datetime.now().strftime('%d/%m/%Y %H:%M:%S')

    def snapshot(self) -> dict:
        with self.lock:
            return {
                'running': self.running,
                'status': self.status,
                'projeto': self.projeto,
                'device': self.device,
                'epoch': self.epoch,
                'epochs_total': self.epochs_total,
                'metrics': dict(self.metrics),
                'log': list(self.log[-40:]),
                'run_dir': self.run_dir,
                'error': self.error,
                'started_at': self.started_at,
                'finished_at': self.finished_at,
            }


training_state = TrainingState()


def _extract_metrics(trainer) -> dict:
    metrics = {}
    try:
        raw = getattr(trainer, 'metrics', None) or {}
        wanted = {
            'metrics/precision(B)': 'precisao',
            'metrics/recall(B)': 'recall',
            'metrics/mAP50(B)': 'mAP50',
            'metrics/mAP50-95(B)': 'mAP50-95',
        }
        for key, label in wanted.items():
            if key in raw:
                metrics[label] = round(float(raw[key]), 4)
    except Exception:
        pass
    try:
        loss_items = getattr(trainer, 'loss_items', None)
        loss_names = getattr(trainer, 'loss_names', None)
        if loss_items is not None and loss_names:
            for name, value in zip(loss_names, loss_items.tolist()):
                metrics[name] = round(float(value), 4)
    except Exception:
        pass
    return metrics


def _resolve_base_model(pp: ProjectPaths, modelo_base: str) -> str:
    if modelo_base == '__ultimo__':
        candidatos = sorted(
            (d for d in pp.runs.iterdir() if d.is_dir() and (d / 'weights' / 'last.pt').exists()),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        ) if pp.runs.exists() else []
        if not candidatos:
            raise ValueError('Nenhum treinamento anterior encontrado para continuar neste modelo.')
        return str(candidatos[0] / 'weights' / 'last.pt')

    permitidos = {'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt'}
    if modelo_base not in permitidos:
        raise ValueError('Modelo base invalido.')
    if modelo_base == 'yolov8n.pt' and LOCAL_BASE_MODEL.exists():
        return str(LOCAL_BASE_MODEL)
    return modelo_base


def _run_training(pp: ProjectPaths, params: dict) -> None:
    run_name = datetime.now().strftime('treino_%Y%m%d_%H%M%S')
    run_dir = pp.runs / run_name
    try:
        classes = load_classes(pp)
        if not classes:
            raise ValueError('Cadastre pelo menos uma classe antes de treinar.')

        train_images = [p for ext in IMAGE_EXTENSIONS for p in pp.images_train.glob(ext)]
        if not train_images:
            raise ValueError('Nao ha imagens anotadas em "Treino". Anote e salve pelo menos uma imagem.')

        val_images = [p for ext in IMAGE_EXTENSIONS for p in pp.images_val.glob(ext)]
        val_subdir = 'images/val' if val_images else 'images/train'
        if not val_images:
            training_state.add_log('Aviso: sem imagens em "Validacao", usando as de treino tambem para validar.')

        data = {
            'path': str(pp.dataset.resolve()),
            'train': 'images/train',
            'val': val_subdir,
            'names': {i: name for i, name in enumerate(classes)},
        }
        data_yaml_path = pp.dataset / 'data.yaml'
        with open(data_yaml_path, 'w', encoding='utf-8') as fh:
            yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)

        base_model = _resolve_base_model(pp, params['modelo_base'])
        training_state.add_log(f'Carregando modelo base: {base_model}')
        model = YOLO(base_model)

        def on_train_start(trainer):
            dispositivo = descrever_dispositivo(trainer)
            training_state.set_device(dispositivo)
            training_state.add_log(
                f"Treinamento iniciado: {params['epocas']} epocas, imgsz={params['imgsz']}, "
                f"batch={params['batch']}, workers={params['workers']}, dispositivo={dispositivo}."
            )

        def on_fit_epoch_end(trainer):
            epoch = int(getattr(trainer, 'epoch', 0)) + 1
            epochs_total = int(getattr(trainer, 'epochs', params['epocas']))
            metrics = _extract_metrics(trainer)
            training_state.update_epoch(epoch, epochs_total, metrics)
            resumo = ', '.join(f'{k}={v}' for k, v in metrics.items())
            training_state.add_log(f'Epoca {epoch}/{epochs_total} - {resumo}')

        model.add_callback('on_train_start', on_train_start)
        model.add_callback('on_fit_epoch_end', on_fit_epoch_end)

        model.train(
            data=str(data_yaml_path),
            epochs=int(params['epocas']),
            imgsz=int(params['imgsz']),
            batch=params['batch'],
            workers=int(params['workers']),
            project=str(pp.runs.resolve()),
            name=run_name,
            exist_ok=True,
            verbose=False,
        )

        training_state.add_log('Treinamento concluido.')
        training_state.finish(run_dir)
    except Exception as exc:
        # O ultralytics as vezes lanca um erro nos graficos finais (results.png,
        # matriz de confusao) depois que o treino em si ja terminou e os pesos
        # ja foram salvos. Nesse caso tratamos como sucesso (com aviso), em vez
        # de descartar um treinamento que na pratica funcionou.
        pesos_existem = (run_dir / 'weights' / 'best.pt').exists() or (run_dir / 'weights' / 'last.pt').exists()
        if pesos_existem:
            training_state.add_log(f'Aviso: erro ao gerar graficos finais, mas o modelo foi salvo normalmente: {exc}')
            training_state.finish(run_dir)
        else:
            training_state.add_log(f'ERRO: {exc}')
            training_state.fail(exc)


PAGE_HTML = """
<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Treinamento de Modelo - Contador IA</title>
  <style>
    :root {
      --bg: #0b1220; --bg-elevated: #0d1526; --card: #141e33; --border: #223052;
      --text: #e7ecf7; --text-muted: #8996b3; --text-faint: #5b6884;
      --accent: #3b82f6; --ok: #22c55e; --err: #ef4444; --warn: #f59e0b;
      --neutral: #64748b; --radius: 14px; --shadow: 0 10px 30px rgba(0,0,0,.35);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Arial, sans-serif;
      background: var(--bg); color: var(--text);
    }
    header.topbar {
      display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 14px 24px;
      background: var(--bg-elevated); border-bottom: 1px solid var(--border); flex-wrap: wrap;
    }
    header.topbar .brand { display: flex; align-items: baseline; gap: 12px; }
    header.topbar h1 { font-size: 16px; margin: 0; }
    header.topbar span { font-size: 12px; color: var(--text-muted); }

    .projeto-row { display: flex; align-items: center; gap: 10px; }
    .projeto-row select {
      background: var(--card); border: 1px solid var(--border); color: var(--text);
      border-radius: 8px; padding: 8px 12px; font-size: 13px; min-width: 220px;
    }

    main.content { max-width: 1240px; margin: 0 auto; padding: 20px 24px 60px; display: flex; flex-direction: column; gap: 20px; }
    .card {
      background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
      padding: 18px; box-shadow: var(--shadow);
    }
    .card h2 {
      margin: 0 0 14px 0; font-size: 12px; text-transform: uppercase;
      letter-spacing: .08em; color: var(--text-muted); font-weight: 700;
    }
    .card-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }
    .card-header h2 { margin: 0; }

    .video-row { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }
    .video-row img { width: 480px; max-width: 100%; border-radius: 10px; background: #000; display: block; }
    .capture-panel { display: flex; flex-direction: column; gap: 10px; min-width: 180px; }
    .hint { font-size: 12px; color: var(--text-muted); }
    .hint.ok { color: var(--ok); }
    .hint.err { color: var(--err); }

    .btn-primary {
      background: var(--accent); border: none; border-radius: 8px; padding: 10px 18px;
      color: #fff; font-size: 13px; font-weight: 600; cursor: pointer;
    }
    .btn-primary:hover { background: #2f6fe0; }
    .btn-primary:disabled { background: var(--neutral); cursor: not-allowed; }
    .btn-secondary {
      background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 9px 16px;
      color: var(--text); font-size: 13px; font-weight: 600; cursor: pointer;
    }
    .btn-secondary:hover { border-color: var(--accent); }

    .chip-list { display: flex; flex-wrap: wrap; gap: 8px; }
    .chip-select {
      border: 1px solid var(--border); background: var(--bg); color: var(--text);
      border-radius: 999px; padding: 6px 12px; font-size: 12px; cursor: pointer;
      border-left: 4px solid var(--chip-color, var(--accent));
    }
    .chip-select.active { background: rgba(59,130,246,.18); border-color: var(--accent); }

    .inline-form { display: flex; gap: 8px; margin-top: 12px; }
    .inline-form input {
      flex: 1; max-width: 260px; background: var(--bg); border: 1px solid var(--border);
      border-radius: 8px; padding: 8px 12px; color: var(--text); font-size: 13px;
    }
    .inline-form button {
      background: var(--card); border: 1px solid var(--border); border-radius: 8px;
      padding: 8px 14px; color: var(--text); font-size: 13px; cursor: pointer;
    }
    .inline-form button:hover { border-color: var(--accent); }

    .tabs { display: flex; gap: 8px; }
    .tab-btn {
      background: var(--bg); border: 1px solid var(--border); color: var(--text-muted);
      border-radius: 8px; padding: 6px 12px; font-size: 12px; cursor: pointer;
    }
    .tab-btn.active { background: rgba(59,130,246,.14); color: var(--accent); border-color: var(--accent); }

    .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 12px; }
    .thumb { background: var(--bg); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; position: relative; }
    .thumb img { width: 100%; height: 110px; object-fit: cover; display: block; background: #000; }
    .thumb-badge {
      position: absolute; top: 6px; right: 6px; font-size: 10px; font-weight: 700;
      padding: 2px 6px; border-radius: 999px; background: rgba(0,0,0,.6);
    }
    .thumb-badge.ok { color: var(--ok); }
    .thumb-badge.warn { color: var(--warn); }
    .thumb-actions { display: flex; flex-direction: column; gap: 4px; padding: 8px; }
    .thumb-btn {
      background: var(--card); border: 1px solid var(--border); color: var(--text);
      border-radius: 6px; padding: 5px 8px; font-size: 11px; cursor: pointer;
    }
    .thumb-btn:hover { border-color: var(--accent); }
    .thumb-btn.danger:hover { border-color: var(--err); color: var(--err); }

    .training-form { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-end; }
    .training-form label {
      display: flex; flex-direction: column; gap: 4px; font-size: 11.5px; color: var(--text-muted);
    }
    .training-form input, .training-form select {
      background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
      padding: 8px 10px; color: var(--text); font-size: 13px; min-width: 140px;
    }

    .training-progress { margin-top: 18px; }
    .progress-bar { background: var(--bg); border: 1px solid var(--border); border-radius: 999px; height: 10px; overflow: hidden; }
    .progress-fill { height: 100%; width: 0%; background: var(--accent); transition: width .3s ease; }
    #progress-text { margin-top: 8px; font-size: 12.5px; color: var(--text-muted); }
    .metrics-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
    .metric-pill {
      font-size: 11px; background: var(--bg); border: 1px solid var(--border); border-radius: 999px;
      padding: 4px 10px; color: var(--text-muted);
    }
    .log-box {
      margin-top: 10px; background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
      padding: 10px; font-size: 11px; color: var(--text-muted); max-height: 180px; overflow-y: auto;
      white-space: pre-wrap;
    }

    .results-list { display: flex; flex-direction: column; gap: 14px; }
    .result-card { background: var(--bg); border: 1px solid var(--border); border-radius: 10px; padding: 12px; }
    .result-header { display: flex; justify-content: space-between; margin-bottom: 8px; }
    .result-images { display: flex; gap: 8px; overflow-x: auto; margin-bottom: 10px; }
    .result-images img { height: 130px; border-radius: 6px; border: 1px solid var(--border); }

    .modal {
      position: fixed; inset: 0; background: rgba(0,0,0,.6); display: none;
      align-items: center; justify-content: center; z-index: 100; padding: 20px;
    }
    .modal-box {
      background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
      max-width: 1100px; width: 100%; max-height: 92vh; overflow-y: auto; box-shadow: var(--shadow);
    }
    .modal-header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 14px 18px; border-bottom: 1px solid var(--border);
    }
    .modal-body { display: flex; gap: 18px; padding: 18px; flex-wrap: wrap; }
    .canvas-wrap { flex: 1 1 480px; }
    #anotacao-canvas { max-width: 100%; border-radius: 8px; background: #000; cursor: crosshair; }
    .modal-side { flex: 0 0 260px; display: flex; flex-direction: column; gap: 6px; }
    .modal-side h3 { font-size: 11px; text-transform: uppercase; color: var(--text-muted); margin: 10px 0 4px; }
    .boxes-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 4px; }
    .boxes-list li { font-size: 12.5px; display: flex; align-items: center; gap: 6px; }
    .link-btn-sm { background: none; border: none; color: var(--accent); font-size: 11px; cursor: pointer; margin-left: auto; }
    .radio-row { display: flex; gap: 14px; font-size: 12.5px; }
    .icon-btn {
      background: var(--bg); border: 1px solid var(--border); color: var(--text-muted);
      border-radius: 8px; width: 28px; height: 28px; cursor: pointer; font-size: 16px; line-height: 1;
    }
    .field-label {
      display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: var(--text-muted); width: 100%;
    }
    .field-label input {
      background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
      padding: 9px 12px; color: var(--text); font-size: 13px;
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand">
      <h1>Treinamento de Modelo</h1>
      <span>{{ titulo }}</span>
    </div>
    <div class="projeto-row">
      <select id="sel-projeto"></select>
      <button id="btn-novo-projeto" class="btn-secondary">+ Novo modelo</button>
    </div>
  </header>

  <main class="content">
    <section class="card">
      <h2>Camera ao vivo</h2>
      <div class="video-row">
        <img src="/video_feed" alt="video" />
        <div class="capture-panel">
          <button id="btn-capturar" class="btn-primary">Capturar frame</button>
          <div id="captura-resultado" class="hint"></div>
        </div>
      </div>
    </section>

    <section class="card">
      <h2>Classes deste modelo</h2>
      <div id="lista-classes" class="chip-list"></div>
      <form id="form-nova-classe" class="inline-form">
        <input type="text" id="nova-classe-nome" placeholder="Nome da nova classe" maxlength="60" autocomplete="off" />
        <button type="submit">Adicionar</button>
      </form>
    </section>

    <section class="card">
      <div class="card-header">
        <h2>Imagens</h2>
        <div class="tabs">
          <button class="tab-btn active" data-tab="pendentes">Pendentes (<span id="count-pendentes">0</span>)</button>
          <button class="tab-btn" data-tab="treino">Treino (<span id="count-treino">0</span>)</button>
          <button class="tab-btn" data-tab="validacao">Validacao (<span id="count-validacao">0</span>)</button>
        </div>
      </div>
      <div id="galeria" class="gallery"></div>
    </section>

    <section class="card">
      <h2>Treinamento</h2>
      <div id="ambiente-info" class="hint" style="margin-bottom: 12px;">Verificando GPU...</div>
      <form id="form-treinamento" class="training-form">
        <label>Epocas
          <input type="number" id="cfg-epocas" value="100" min="1" max="2000" />
        </label>
        <label>Tamanho da imagem
          <input type="number" id="cfg-imgsz" value="640" min="32" max="1920" step="32" />
        </label>
        <label>Batch
          <select id="cfg-batch">
            <option value="auto">Automatico</option>
            <option value="4">4</option>
            <option value="8" selected>8</option>
            <option value="16">16</option>
            <option value="32">32</option>
          </select>
        </label>
        <label>Workers
          <input type="number" id="cfg-workers" value="4" min="0" max="16" />
        </label>
        <label>Modelo base
          <select id="cfg-modelo">
            <option value="yolov8n.pt" selected>YOLOv8n (rapido, leve)</option>
            <option value="yolov8s.pt">YOLOv8s (equilibrado, precisa internet)</option>
            <option value="yolov8m.pt">YOLOv8m (mais preciso, precisa internet)</option>
            <option value="__ultimo__">Continuar do ultimo treinamento deste modelo</option>
          </select>
        </label>
        <button type="submit" id="btn-iniciar-treino" class="btn-primary">Iniciar treinamento</button>
      </form>
      <p class="hint" style="margin-top: 8px;">
        "Workers" controla quantos processos carregam/preparam imagens em paralelo
        durante o treino. Valores mais altos usam mais RAM e podem ser mais rapidos;
        se o treino travar/crashar (comum com GPU de pouca memoria), reduza para 2, 1 ou 0.
      </p>

      <div id="treino-progresso" class="training-progress" style="display:none;">
        <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
        <div id="progress-text">-</div>
        <div id="device-info" class="hint"></div>
        <div id="metrics-row" class="metrics-row"></div>
        <pre id="log-box" class="log-box"></pre>
      </div>
    </section>

    <section class="card">
      <h2>Resultados</h2>
      <div id="lista-resultados" class="results-list"><p class="hint">Nenhum treinamento concluido ainda.</p></div>
    </section>
  </main>

  <div id="modal-anotacao" class="modal">
    <div class="modal-box">
      <div class="modal-header">
        <strong id="modal-titulo">Anotar imagem</strong>
        <button id="modal-fechar" class="icon-btn">&times;</button>
      </div>
      <div class="modal-body">
        <div class="canvas-wrap">
          <canvas id="anotacao-canvas"></canvas>
        </div>
        <div class="modal-side">
          <h3>Classe ativa (usada ao desenhar)</h3>
          <div id="modal-classes" class="chip-list"></div>
          <h3>Caixas desenhadas</h3>
          <ul id="lista-caixas" class="boxes-list"><li class="hint">Nenhuma caixa desenhada.</li></ul>
          <h3>Salvar como</h3>
          <div class="radio-row">
            <label><input type="radio" name="destino" value="treino" checked /> Treino</label>
            <label><input type="radio" name="destino" value="validacao" /> Validacao</label>
          </div>
          <div style="margin-top: 12px;">
            <button id="btn-salvar-anotacao" class="btn-primary">Salvar anotacao</button>
          </div>
          <div id="anotacao-resultado" class="hint"></div>
        </div>
      </div>
    </div>
  </div>

  <div id="modal-novo-projeto" class="modal">
    <div class="modal-box" style="max-width: 440px;">
      <div class="modal-header">
        <strong>Criar novo modelo</strong>
        <button id="modal-novo-projeto-fechar" class="icon-btn">&times;</button>
      </div>
      <div class="modal-body" style="flex-direction: column;">
        <label class="field-label">Nome do modelo
          <input type="text" id="novo-projeto-nome" placeholder="Ex: Treinamento Mercados" maxlength="80" autocomplete="off" />
        </label>
        <label class="field-label" style="margin-top: 10px;">Classes iniciais (opcional, separadas por virgula)
          <input type="text" id="novo-projeto-classes" placeholder="Ex: PULL, ESTANTE" autocomplete="off" />
        </label>
        <button id="btn-criar-projeto" class="btn-primary" style="margin-top: 14px;">Criar modelo</button>
        <div id="novo-projeto-resultado" class="hint"></div>
      </div>
    </div>
  </div>

  <script>
    const PALETTE = ['#22c55e', '#f59e0b', '#3b82f6', '#ef4444', '#a855f7', '#14b8a6', '#eab308', '#ec4899'];
    const state = { classes: [], activeClassIndex: 0, currentImage: null, boxes: [], drawing: false };
    let abaAtiva = 'pendentes';
    let projetoAtual = localStorage.getItem('treinamento_projeto_atual') || '';

    function qp(extra) {
      const params = new URLSearchParams(Object.assign({ projeto: projetoAtual }, extra || {}));
      return params.toString();
    }

    async function carregarAmbiente() {
      const el = document.getElementById('ambiente-info');
      try {
        const res = await fetch('/ambiente');
        const info = await res.json();
        if (info.gpu_disponivel) {
          el.textContent = `GPU detectada: ${info.gpu_nome} (${info.gpu_memoria_gb} GB) - sera usada automaticamente no treinamento.`;
          el.className = 'hint ok';
        } else {
          el.textContent = 'Nenhuma GPU NVIDIA/CUDA detectada - o treinamento vai rodar na CPU (mais lento).';
          el.className = 'hint';
        }
      } catch (e) {
        el.textContent = 'Nao foi possivel verificar GPU.';
        el.className = 'hint';
      }
    }

    async function carregarProjetos() {
      const res = await fetch('/projetos');
      const lista = await res.json();
      const sel = document.getElementById('sel-projeto');
      if (!lista.length) {
        sel.innerHTML = '<option value="">Nenhum modelo ainda - crie um</option>';
        projetoAtual = '';
        atualizarEstadoSemProjeto();
        return;
      }
      if (!projetoAtual || !lista.includes(projetoAtual)) projetoAtual = lista[0];
      sel.innerHTML = lista.map((p) => `<option value="${p}" ${p === projetoAtual ? 'selected' : ''}>${p}</option>`).join('');
      localStorage.setItem('treinamento_projeto_atual', projetoAtual);
      await atualizarTudoDoProjeto();
    }

    function atualizarEstadoSemProjeto() {
      document.getElementById('lista-classes').innerHTML = '<span class="hint">Crie um modelo primeiro.</span>';
      document.getElementById('galeria').innerHTML = '<p class="hint">Crie um modelo primeiro.</p>';
      document.getElementById('lista-resultados').innerHTML = '<p class="hint">Crie um modelo primeiro.</p>';
      ['count-pendentes', 'count-treino', 'count-validacao'].forEach((id) => { document.getElementById(id).textContent = '0'; });
    }

    document.getElementById('sel-projeto').addEventListener('change', async (ev) => {
      projetoAtual = ev.target.value;
      localStorage.setItem('treinamento_projeto_atual', projetoAtual);
      await atualizarTudoDoProjeto();
    });

    async function atualizarTudoDoProjeto() {
      if (!projetoAtual) { atualizarEstadoSemProjeto(); return; }
      await carregarClasses();
      await carregarImagens();
      await carregarResultados();
      await atualizarStatusTreino();
    }

    const modalNovoProjeto = document.getElementById('modal-novo-projeto');
    document.getElementById('btn-novo-projeto').addEventListener('click', () => {
      document.getElementById('novo-projeto-nome').value = '';
      document.getElementById('novo-projeto-classes').value = '';
      document.getElementById('novo-projeto-resultado').textContent = '';
      modalNovoProjeto.style.display = 'flex';
    });
    document.getElementById('modal-novo-projeto-fechar').addEventListener('click', () => { modalNovoProjeto.style.display = 'none'; });

    document.getElementById('btn-criar-projeto').addEventListener('click', async () => {
      const nome = document.getElementById('novo-projeto-nome').value.trim();
      const classesRaw = document.getElementById('novo-projeto-classes').value.trim();
      const resultEl = document.getElementById('novo-projeto-resultado');
      if (!nome) { resultEl.textContent = 'Digite um nome.'; resultEl.className = 'hint err'; return; }
      const classes_iniciais = classesRaw ? classesRaw.split(',').map((c) => c.trim()).filter(Boolean) : [];
      const res = await fetch('/projetos', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ nome, classes_iniciais }),
      });
      const data = await res.json();
      if (data.ok) {
        resultEl.textContent = 'Modelo criado!';
        resultEl.className = 'hint ok';
        projetoAtual = data.projeto;
        setTimeout(async () => { modalNovoProjeto.style.display = 'none'; await carregarProjetos(); }, 400);
      } else {
        resultEl.textContent = data.erro || 'Falha ao criar modelo.';
        resultEl.className = 'hint err';
      }
    });

    async function carregarClasses() {
      const res = await fetch(`/classes?${qp()}`);
      state.classes = await res.json();
      renderClassChips();
    }

    function renderClassChips() {
      const html = state.classes.length
        ? state.classes.map((nome, i) => `
            <button type="button" class="chip-select ${i === state.activeClassIndex ? 'active' : ''}" data-idx="${i}" style="--chip-color:${PALETTE[i % PALETTE.length]}">${nome}</button>
          `).join('')
        : '<span class="hint">Nenhuma classe cadastrada ainda.</span>';
      const principal = document.getElementById('lista-classes');
      const modal = document.getElementById('modal-classes');
      if (principal) principal.innerHTML = html;
      if (modal) modal.innerHTML = html;
      document.querySelectorAll('.chip-select').forEach((btn) => {
        btn.addEventListener('click', () => {
          state.activeClassIndex = parseInt(btn.getAttribute('data-idx'), 10);
          renderClassChips();
        });
      });
    }

    document.getElementById('form-nova-classe').addEventListener('submit', async (ev) => {
      ev.preventDefault();
      if (!projetoAtual) { alert('Crie um modelo primeiro.'); return; }
      const input = document.getElementById('nova-classe-nome');
      const nome = input.value.trim();
      if (!nome) return;
      const res = await fetch('/classes', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ projeto: projetoAtual, nome }),
      });
      const data = await res.json();
      if (data.ok) { input.value = ''; await carregarClasses(); }
      else alert(data.erro || 'Falha ao adicionar classe.');
    });

    document.getElementById('btn-capturar').addEventListener('click', async () => {
      if (!projetoAtual) { alert('Crie um modelo primeiro.'); return; }
      const resultEl = document.getElementById('captura-resultado');
      resultEl.textContent = 'Capturando...';
      resultEl.className = 'hint';
      try {
        const res = await fetch('/capturar', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ projeto: projetoAtual }),
        });
        const data = await res.json();
        if (data.ok) {
          resultEl.textContent = `Salvo: ${data.arquivo}`;
          resultEl.className = 'hint ok';
          carregarImagens();
        } else {
          resultEl.textContent = data.erro || 'Falha ao capturar.';
          resultEl.className = 'hint err';
        }
      } catch (_) {
        resultEl.textContent = 'Falha ao contatar o servidor.';
        resultEl.className = 'hint err';
      }
    });

    async function carregarImagens() {
      if (!projetoAtual) return;
      const res = await fetch(`/imagens?${qp()}`);
      const data = await res.json();
      document.getElementById('count-pendentes').textContent = data.pendentes.length;
      document.getElementById('count-treino').textContent = data.treino.length;
      document.getElementById('count-validacao').textContent = data.validacao.length;
      renderGaleria(data[abaAtiva] || []);
    }

    function renderGaleria(itens) {
      const wrap = document.getElementById('galeria');
      if (!itens.length) {
        wrap.innerHTML = '<p class="hint">Nenhuma imagem aqui ainda.</p>';
        return;
      }
      wrap.innerHTML = itens.map((item) => `
        <div class="thumb" data-nome="${item.nome}">
          <img src="/imagem?${qp({ pasta: abaAtiva, nome: item.nome })}" loading="lazy" />
          ${item.anotado !== undefined ? `<span class="thumb-badge ${item.anotado ? 'ok' : 'warn'}">${item.anotado ? 'Anotada' : 'Sem anotacao'}</span>` : ''}
          <div class="thumb-actions">
            <button class="thumb-btn" data-acao="anotar">Anotar</button>
            ${abaAtiva !== 'pendentes' ? `<button class="thumb-btn" data-acao="mover">${abaAtiva === 'treino' ? 'Mover p/ validacao' : 'Mover p/ treino'}</button>` : ''}
            <button class="thumb-btn danger" data-acao="excluir">Excluir</button>
          </div>
        </div>
      `).join('');

      wrap.querySelectorAll('.thumb').forEach((el) => {
        const nome = el.getAttribute('data-nome');
        el.querySelector('[data-acao="anotar"]').addEventListener('click', () => abrirAnotacao(abaAtiva, nome));
        const moverBtn = el.querySelector('[data-acao="mover"]');
        if (moverBtn) moverBtn.addEventListener('click', () => moverImagem(nome));
        el.querySelector('[data-acao="excluir"]').addEventListener('click', () => excluirImagem(nome));
      });
    }

    document.querySelectorAll('.tab-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        abaAtiva = btn.getAttribute('data-tab');
        carregarImagens();
      });
    });

    async function moverImagem(nome) {
      const destino = abaAtiva === 'treino' ? 'validacao' : 'treino';
      await fetch('/imagens/mover', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ projeto: projetoAtual, nome, origem: abaAtiva, destino }),
      });
      carregarImagens();
    }

    async function excluirImagem(nome) {
      if (!confirm(`Excluir ${nome}?`)) return;
      await fetch('/imagens/excluir', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ projeto: projetoAtual, nome, pasta: abaAtiva }),
      });
      carregarImagens();
    }

    const modal = document.getElementById('modal-anotacao');
    const canvas = document.getElementById('anotacao-canvas');
    const ctx = canvas.getContext('2d');
    let imagemCarregada = null;

    async function abrirAnotacao(pasta, nome) {
      state.currentImage = { pasta, nome };
      state.boxes = [];
      document.getElementById('modal-titulo').textContent = `Anotar: ${nome}`;
      document.getElementById('anotacao-resultado').textContent = '';
      renderClassChips();
      renderListaCaixas();

      if (pasta !== 'pendentes') {
        try {
          const res = await fetch(`/anotacao?${qp({ pasta, nome })}`);
          if (res.ok) {
            const data = await res.json();
            state.boxes = data.caixas || [];
          }
        } catch (_) {}
      }

      const img = new Image();
      img.onload = () => {
        imagemCarregada = img;
        const maxW = Math.min(760, window.innerWidth - 340);
        const scale = Math.min(1, maxW / img.naturalWidth);
        canvas.width = Math.round(img.naturalWidth * scale);
        canvas.height = Math.round(img.naturalHeight * scale);
        renderListaCaixas();
      };
      img.src = `/imagem?${qp({ pasta, nome })}`;

      modal.style.display = 'flex';
    }

    document.getElementById('modal-fechar').addEventListener('click', () => { modal.style.display = 'none'; });

    function desenharCanvas() {
      if (!imagemCarregada) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(imagemCarregada, 0, 0, canvas.width, canvas.height);
      state.boxes.forEach((box) => {
        const cor = PALETTE[box.classe % PALETTE.length];
        const x = box.x * canvas.width, y = box.y * canvas.height;
        const w = box.w * canvas.width, h = box.h * canvas.height;
        ctx.strokeStyle = cor; ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);
        const nomeClasse = state.classes[box.classe] || '?';
        ctx.font = '11px sans-serif';
        const labelW = ctx.measureText(nomeClasse).width + 8;
        ctx.fillStyle = cor;
        ctx.fillRect(x, Math.max(0, y - 16), labelW, 16);
        ctx.fillStyle = '#0b1220';
        ctx.fillText(nomeClasse, x + 4, Math.max(11, y - 4));
      });
      if (state.drawing) {
        ctx.strokeStyle = PALETTE[state.activeClassIndex % PALETTE.length];
        ctx.setLineDash([4, 3]);
        ctx.strokeRect(state.drawX, state.drawY, state.drawW, state.drawH);
        ctx.setLineDash([]);
      }
    }

    canvas.addEventListener('mousedown', (ev) => {
      if (!state.classes.length) { alert('Cadastre uma classe antes de desenhar.'); return; }
      const rect = canvas.getBoundingClientRect();
      state.drawing = true;
      state.startX = ev.clientX - rect.left;
      state.startY = ev.clientY - rect.top;
      state.drawX = state.startX; state.drawY = state.startY; state.drawW = 0; state.drawH = 0;
    });

    canvas.addEventListener('mousemove', (ev) => {
      if (!state.drawing) return;
      const rect = canvas.getBoundingClientRect();
      const curX = ev.clientX - rect.left, curY = ev.clientY - rect.top;
      state.drawX = Math.min(state.startX, curX);
      state.drawY = Math.min(state.startY, curY);
      state.drawW = Math.abs(curX - state.startX);
      state.drawH = Math.abs(curY - state.startY);
      desenharCanvas();
    });

    window.addEventListener('mouseup', () => {
      if (!state.drawing) return;
      state.drawing = false;
      if (state.drawW > 4 && state.drawH > 4 && canvas.width > 0) {
        state.boxes.push({
          classe: state.activeClassIndex,
          x: state.drawX / canvas.width, y: state.drawY / canvas.height,
          w: state.drawW / canvas.width, h: state.drawH / canvas.height,
        });
        renderListaCaixas();
      } else {
        desenharCanvas();
      }
    });

    function renderListaCaixas() {
      const ul = document.getElementById('lista-caixas');
      if (!state.boxes.length) {
        ul.innerHTML = '<li class="hint">Nenhuma caixa desenhada.</li>';
      } else {
        ul.innerHTML = state.boxes.map((box, i) => `
          <li>
            <span style="color:${PALETTE[box.classe % PALETTE.length]}">&#9632;</span>
            ${state.classes[box.classe] || '?'}
            <button type="button" data-idx="${i}" class="link-btn-sm">remover</button>
          </li>
        `).join('');
        ul.querySelectorAll('button[data-idx]').forEach((btn) => {
          btn.addEventListener('click', () => {
            state.boxes.splice(parseInt(btn.getAttribute('data-idx'), 10), 1);
            renderListaCaixas();
          });
        });
      }
      desenharCanvas();
    }

    document.getElementById('btn-salvar-anotacao').addEventListener('click', async () => {
      const destino = document.querySelector('input[name="destino"]:checked').value;
      const resultEl = document.getElementById('anotacao-resultado');
      const res = await fetch('/anotar', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          projeto: projetoAtual,
          pasta: state.currentImage.pasta,
          nome: state.currentImage.nome,
          destino,
          caixas: state.boxes.map((b) => ({ classe: b.classe, x: b.x + b.w / 2, y: b.y + b.h / 2, w: b.w, h: b.h })),
        }),
      });
      const data = await res.json();
      if (data.ok) {
        resultEl.textContent = 'Salvo!';
        resultEl.className = 'hint ok';
        setTimeout(() => { modal.style.display = 'none'; carregarImagens(); }, 500);
      } else {
        resultEl.textContent = data.erro || 'Falha ao salvar.';
        resultEl.className = 'hint err';
      }
    });

    let treinoPollTimer = null;

    document.getElementById('form-treinamento').addEventListener('submit', async (ev) => {
      ev.preventDefault();
      if (!projetoAtual) { alert('Crie um modelo primeiro.'); return; }
      const btn = document.getElementById('btn-iniciar-treino');
      const epocas = document.getElementById('cfg-epocas').value;
      const imgsz = document.getElementById('cfg-imgsz').value;
      const batch = document.getElementById('cfg-batch').value;
      const workers = document.getElementById('cfg-workers').value;
      const modelo_base = document.getElementById('cfg-modelo').value;

      const res = await fetch('/treinamento/iniciar', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ projeto: projetoAtual, epocas, imgsz, batch, workers, modelo_base }),
      });
      const data = await res.json();
      if (!data.ok) { alert(data.erro || 'Falha ao iniciar treinamento.'); return; }
      document.getElementById('treino-progresso').style.display = 'block';
      btn.disabled = true;
      clearInterval(treinoPollTimer);
      treinoPollTimer = setInterval(atualizarStatusTreino, 2000);
      atualizarStatusTreino();
    });

    async function atualizarStatusTreino() {
      if (!projetoAtual) return;
      const res = await fetch('/treinamento/status');
      const s = await res.json();
      const btn = document.getElementById('btn-iniciar-treino');
      const fill = document.getElementById('progress-fill');
      const text = document.getElementById('progress-text');
      const deviceInfo = document.getElementById('device-info');
      const metricsRow = document.getElementById('metrics-row');
      const logBox = document.getElementById('log-box');

      const ehEsteProjeto = s.projeto === projetoAtual;

      if (s.status !== 'ocioso' && ehEsteProjeto) document.getElementById('treino-progresso').style.display = 'block';

      const pct = s.epochs_total > 0 ? Math.round((s.epoch / s.epochs_total) * 100) : 0;
      if (ehEsteProjeto) {
        fill.style.width = `${pct}%`;
        text.textContent = s.status === 'rodando' ? `Epoca ${s.epoch} de ${s.epochs_total} (${pct}%)`
          : s.status === 'concluido' ? 'Treinamento concluido!'
          : s.status === 'erro' ? `Erro: ${s.error}` : 'Aguardando...';
        deviceInfo.textContent = s.device ? `Rodando em: ${s.device}` : '';
        metricsRow.innerHTML = Object.entries(s.metrics || {}).map(([k, v]) => `<span class="metric-pill">${k}: ${v}</span>`).join('');
        logBox.textContent = (s.log || []).join('\\n');
        logBox.scrollTop = logBox.scrollHeight;
      } else if (s.running) {
        text.textContent = `Ha um treinamento em andamento no modelo "${s.projeto}". Aguarde terminar para treinar este.`;
      }

      btn.disabled = s.running;

      if (ehEsteProjeto && (s.status === 'concluido' || s.status === 'erro')) {
        clearInterval(treinoPollTimer);
        if (s.status === 'concluido') carregarResultados();
      }
    }

    async function carregarResultados() {
      if (!projetoAtual) return;
      const res = await fetch(`/treinamento/resultados?${qp()}`);
      const lista = await res.json();
      const wrap = document.getElementById('lista-resultados');
      if (!lista.length) { wrap.innerHTML = '<p class="hint">Nenhum treinamento concluido ainda.</p>'; return; }
      wrap.innerHTML = lista.map((r) => `
        <div class="result-card">
          <div class="result-header"><strong>${r.nome}</strong><span class="hint">${r.modificado}</span></div>
          <div class="result-images">${r.imagens.map((img) => `<img src="/treinamento/resultados/${r.nome}/${img}?${qp()}" loading="lazy" />`).join('')}</div>
          ${r.tem_modelo ? `<button class="btn-primary" data-run="${r.nome}">Usar este modelo</button>` : '<span class="hint">Sem best.pt gerado</span>'}
        </div>
      `).join('');
      wrap.querySelectorAll('button[data-run]').forEach((btn) => {
        btn.addEventListener('click', async () => {
          const run = btn.getAttribute('data-run');
          const destinoPadrao = '../runs/detect/train/weights/best.pt';
          const destino = prompt('Copiar o modelo treinado para qual caminho?', destinoPadrao);
          if (destino === null) return;
          const res = await fetch(`/treinamento/aplicar/${run}`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ projeto: projetoAtual, destino }),
          });
          const data = await res.json();
          alert(data.ok ? `Modelo copiado para ${data.destino}. Reinicie o programa que usa esse modelo.` : (data.erro || 'Falha ao aplicar.'));
        });
      });
    }

    carregarAmbiente();
    carregarProjetos();
  </script>
</body>
</html>
"""


def create_app(stream: CameraStream, titulo: str) -> Flask:
    app = Flask(__name__)

    @app.get('/')
    def index():
        return render_template_string(PAGE_HTML, titulo=titulo)

    @app.get('/projetos')
    def listar_projetos():
        return jsonify(list_projetos())

    @app.post('/projetos')
    def criar_projeto():
        data = request.get_json(silent=True) or {}
        try:
            pp = create_project(str(data.get('nome', '')), data.get('classes_iniciais'))
        except ValueError as exc:
            return jsonify({'ok': False, 'erro': str(exc)}), 400
        return jsonify({'ok': True, 'projeto': pp.nome})

    @app.get('/video_feed')
    def video_feed():
        def generate():
            target_fps = 10
            interval = 1.0 / target_fps
            last_sent = 0.0
            while True:
                now = time.time()
                if now - last_sent < interval:
                    time.sleep(0.02)
                    continue
                frame = stream.get_frame()
                if frame is None:
                    time.sleep(0.02)
                    continue
                ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if not ok:
                    continue
                last_sent = now
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.post('/capturar')
    def capturar():
        data = request.get_json(silent=True) or {}
        pp = get_project(data.get('projeto', ''))
        frame = stream.get_frame()
        if frame is None:
            return jsonify({'ok': False, 'erro': 'Sem frame disponivel da camera ainda.'}), 503
        nome = f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        destino = pp.capturas / nome
        if not cv2.imwrite(str(destino), frame):
            return jsonify({'ok': False, 'erro': 'Falha ao salvar a imagem.'}), 500
        return jsonify({'ok': True, 'arquivo': nome})

    def _listar(pasta_dir: Path, label_dir=None):
        itens = []
        arquivos = []
        for ext in IMAGE_EXTENSIONS:
            arquivos.extend(pasta_dir.glob(ext))
        for p in sorted(arquivos, key=lambda f: f.stat().st_mtime, reverse=True):
            info = {'nome': p.name}
            if label_dir is not None:
                lbl = label_dir / (p.stem + '.txt')
                info['anotado'] = lbl.exists() and lbl.stat().st_size > 0
            itens.append(info)
        return itens

    @app.get('/imagens')
    def listar_imagens():
        pp = get_project(request.args.get('projeto', ''))
        return jsonify({
            'pendentes': _listar(pp.capturas),
            'treino': _listar(pp.images_train, pp.labels_train),
            'validacao': _listar(pp.images_val, pp.labels_val),
        })

    def _resolve_image_path(pp: ProjectPaths, pasta: str, nome: str):
        mapping = {'pendentes': pp.capturas, 'treino': pp.images_train, 'validacao': pp.images_val}
        base = mapping.get(pasta)
        if base is None:
            return None
        return base / safe_name(nome)

    @app.get('/imagem')
    def servir_imagem():
        pp = get_project(request.args.get('projeto', ''))
        path = _resolve_image_path(pp, request.args.get('pasta', ''), request.args.get('nome', ''))
        if path is None or not path.exists() or not path.is_file():
            abort(404)
        return send_file(path)

    @app.get('/anotacao')
    def obter_anotacao():
        pp = get_project(request.args.get('projeto', ''))
        pasta = request.args.get('pasta', '')
        nome = safe_name(request.args.get('nome', ''))
        mapping = {'treino': pp.labels_train, 'validacao': pp.labels_val}
        lbl_dir = mapping.get(pasta)
        caixas = []
        if lbl_dir is not None:
            lbl_path = lbl_dir / (Path(nome).stem + '.txt')
            if lbl_path.exists():
                for linha in lbl_path.read_text(encoding='utf-8').splitlines():
                    partes = linha.strip().split()
                    if len(partes) != 5:
                        continue
                    try:
                        cls_id = int(partes[0])
                        xc, yc, w, h = (float(v) for v in partes[1:])
                    except ValueError:
                        continue
                    caixas.append({'classe': cls_id, 'x': xc - w / 2, 'y': yc - h / 2, 'w': w, 'h': h})
        return jsonify({'caixas': caixas})

    @app.post('/anotar')
    def anotar():
        data = request.get_json(silent=True) or {}
        pp = get_project(data.get('projeto', ''))
        pasta_origem = data.get('pasta', 'pendentes')
        nome = safe_name(data.get('nome', ''))
        destino = data.get('destino', 'treino')
        caixas = data.get('caixas', [])

        origem_map = {
            'pendentes': (pp.capturas, None),
            'treino': (pp.images_train, pp.labels_train),
            'validacao': (pp.images_val, pp.labels_val),
        }
        destino_map = {'treino': (pp.images_train, pp.labels_train), 'validacao': (pp.images_val, pp.labels_val)}

        if destino not in destino_map or pasta_origem not in origem_map or not nome:
            return jsonify({'ok': False, 'erro': 'Parametros invalidos.'}), 400

        origem_img_dir, origem_lbl_dir = origem_map[pasta_origem]
        origem_img = origem_img_dir / nome
        if not origem_img.exists():
            return jsonify({'ok': False, 'erro': 'Imagem nao encontrada.'}), 404

        classes = load_classes(pp)
        linhas = []
        for caixa in caixas:
            try:
                cls_id = int(caixa['classe'])
                x = float(caixa['x'])
                y = float(caixa['y'])
                w = float(caixa['w'])
                h = float(caixa['h'])
            except (KeyError, TypeError, ValueError):
                continue
            if not (0 <= cls_id < len(classes)):
                continue
            x = min(max(x, 0.0), 1.0)
            y = min(max(y, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)
            linhas.append(f'{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}')

        dest_img_dir, dest_lbl_dir = destino_map[destino]
        dest_img_path = dest_img_dir / nome
        dest_lbl_path = dest_lbl_dir / (Path(nome).stem + '.txt')

        if origem_img.resolve() != dest_img_path.resolve():
            shutil.move(str(origem_img), str(dest_img_path))
        if origem_lbl_dir is not None:
            origem_lbl = origem_lbl_dir / (Path(nome).stem + '.txt')
            if origem_lbl.exists() and origem_lbl.resolve() != dest_lbl_path.resolve():
                origem_lbl.unlink(missing_ok=True)

        dest_lbl_path.write_text('\n'.join(linhas) + ('\n' if linhas else ''), encoding='utf-8')
        return jsonify({'ok': True})

    @app.post('/imagens/mover')
    def mover_imagem():
        data = request.get_json(silent=True) or {}
        pp = get_project(data.get('projeto', ''))
        nome = safe_name(data.get('nome', ''))
        origem = data.get('origem')
        destino = data.get('destino')
        mapping = {'treino': (pp.images_train, pp.labels_train), 'validacao': (pp.images_val, pp.labels_val)}
        if origem not in mapping or destino not in mapping or origem == destino or not nome:
            return jsonify({'ok': False, 'erro': 'Parametros invalidos.'}), 400

        origem_img_dir, origem_lbl_dir = mapping[origem]
        destino_img_dir, destino_lbl_dir = mapping[destino]
        origem_img = origem_img_dir / nome
        if not origem_img.exists():
            return jsonify({'ok': False, 'erro': 'Imagem nao encontrada.'}), 404

        shutil.move(str(origem_img), str(destino_img_dir / nome))
        origem_lbl = origem_lbl_dir / (Path(nome).stem + '.txt')
        if origem_lbl.exists():
            shutil.move(str(origem_lbl), str(destino_lbl_dir / (Path(nome).stem + '.txt')))
        return jsonify({'ok': True})

    @app.post('/imagens/excluir')
    def excluir_imagem():
        data = request.get_json(silent=True) or {}
        pp = get_project(data.get('projeto', ''))
        nome = safe_name(data.get('nome', ''))
        pasta = data.get('pasta')
        mapping = {
            'pendentes': (pp.capturas, None),
            'treino': (pp.images_train, pp.labels_train),
            'validacao': (pp.images_val, pp.labels_val),
        }
        if pasta not in mapping or not nome:
            return jsonify({'ok': False, 'erro': 'Parametros invalidos.'}), 400
        img_dir, lbl_dir = mapping[pasta]
        img_path = img_dir / nome
        if img_path.exists():
            img_path.unlink()
        if lbl_dir is not None:
            (lbl_dir / (Path(nome).stem + '.txt')).unlink(missing_ok=True)
        return jsonify({'ok': True})

    @app.get('/classes')
    def get_classes():
        pp = get_project(request.args.get('projeto', ''))
        return jsonify(load_classes(pp))

    @app.post('/classes')
    def add_class():
        data = request.get_json(silent=True) or {}
        pp = get_project(data.get('projeto', ''))
        nome = str(data.get('nome', '')).strip()
        if not nome:
            return jsonify({'ok': False, 'erro': 'Nome vazio.'}), 400
        classes = load_classes(pp)
        if nome in classes:
            return jsonify({'ok': False, 'erro': 'Classe ja existe.'}), 400
        classes.append(nome)
        save_classes(pp, classes)
        return jsonify({'ok': True, 'classes': classes})

    @app.post('/treinamento/iniciar')
    def iniciar_treinamento():
        if training_state.running:
            return jsonify({'ok': False, 'erro': f'Ja existe um treinamento em andamento (modelo "{training_state.projeto}").'}), 409
        data = request.get_json(silent=True) or {}
        pp = get_project(data.get('projeto', ''))
        try:
            epocas = int(data.get('epocas', 100))
            imgsz = int(data.get('imgsz', 640))
            batch_raw = data.get('batch', 8)
            batch = -1 if str(batch_raw).lower() == 'auto' else int(batch_raw)
            modelo_base = str(data.get('modelo_base', 'yolov8n.pt'))
            workers = int(data.get('workers', 4))
        except (TypeError, ValueError):
            return jsonify({'ok': False, 'erro': 'Parametros de treinamento invalidos.'}), 400

        if epocas < 1 or imgsz < 32 or workers < 0:
            return jsonify({'ok': False, 'erro': 'Epocas, tamanho de imagem ou workers invalidos.'}), 400

        training_state.reset_for_start(pp.nome, epocas)
        params = {'epocas': epocas, 'imgsz': imgsz, 'batch': batch, 'modelo_base': modelo_base, 'workers': workers}
        threading.Thread(target=_run_training, args=(pp, params), daemon=True).start()
        return jsonify({'ok': True})

    @app.get('/treinamento/status')
    def status_treinamento():
        return jsonify(training_state.snapshot())

    @app.get('/ambiente')
    def ambiente():
        return jsonify(detectar_ambiente())

    @app.get('/treinamento/resultados')
    def listar_resultados():
        pp = get_project(request.args.get('projeto', ''))
        resultados = []
        if pp.runs.exists():
            for d in sorted((p for p in pp.runs.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
                best = d / 'weights' / 'best.pt'
                imagens = sorted(d.glob('*.png')) + sorted(d.glob('*.jpg'))
                resultados.append({
                    'nome': d.name,
                    'tem_modelo': best.exists(),
                    'modificado': datetime.fromtimestamp(d.stat().st_mtime).strftime('%d/%m/%Y %H:%M'),
                    'imagens': [f.name for f in imagens],
                })
        return jsonify(resultados)

    @app.get('/treinamento/resultados/<run>/<path:arquivo>')
    def arquivo_resultado(run, arquivo):
        pp = get_project(request.args.get('projeto', ''))
        path = pp.runs / safe_name(run) / safe_name(arquivo)
        if not path.exists() or not path.is_file():
            abort(404)
        return send_file(path)

    @app.post('/treinamento/aplicar/<run>')
    def aplicar_resultado(run):
        data = request.get_json(silent=True) or {}
        pp = get_project(data.get('projeto', ''))
        best = pp.runs / safe_name(run) / 'weights' / 'best.pt'
        if not best.exists():
            return jsonify({'ok': False, 'erro': 'Modelo nao encontrado para essa execucao.'}), 404
        destino_raw = str(data.get('destino') or '../runs/detect/train/weights/best.pt').strip()
        destino = Path(destino_raw)
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, destino)
        return jsonify({'ok': True, 'destino': str(destino)})

    return app


def main() -> None:
    parser = load_config(CONFIG_FILE)
    rtsp_url = load_rtsp_url(parser)

    stream = CameraStream(rtsp_url)
    if not stream.is_opened():
        raise RuntimeError('Nao foi possivel abrir o stream RTSP. Verifique config/camera.')
    stream.start()

    host = parser.get('web', 'host', fallback='0.0.0.0')
    port = parser.getint('web', 'porta', fallback=5011)
    titulo = parser.get('web', 'titulo', fallback='Treinamento de Modelo')

    app = create_app(stream, titulo)
    try:
        print(f'Web de treinamento rodando em http://{host}:{port}')
        app.run(host=host, port=port, debug=False, threaded=True)
    finally:
        stream.stop()


if __name__ == '__main__':
    main()
