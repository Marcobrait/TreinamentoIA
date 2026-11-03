# python.exe -m venv .venv
# .\.venv\Scripts\Activate.ps1
# python -m pip install --upgrade pip
# python -m pip install -r requirements.txt

import base64
import configparser
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
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
        self.task_file = self.dataset / 'task.txt'
        self.runs = self.root / 'runs_treinamento'

    def ensure(self) -> None:
        for d in (self.capturas, self.images_train, self.images_val, self.labels_train, self.labels_val, self.runs):
            d.mkdir(parents=True, exist_ok=True)
        if not self.classes_file.exists():
            self.classes_file.write_text('', encoding='utf-8')

    def exists(self) -> bool:
        return self.root.exists()


def load_project_task(pp: ProjectPaths) -> str:
    if pp.task_file.exists():
        conteudo = pp.task_file.read_text(encoding='utf-8').strip()
        if conteudo == 'segment':
            return 'segment'
    return 'detect'


def save_project_task(pp: ProjectPaths, task: str) -> None:
    pp.task_file.write_text(task, encoding='utf-8')


def list_projetos() -> list:
    if not PROJETOS_DIR.exists():
        return []
    return sorted(d.name for d in PROJETOS_DIR.iterdir() if d.is_dir())


def get_project(nome: str) -> ProjectPaths:
    slug = slugify_projeto(nome) or DEFAULT_PROJETO
    pp = ProjectPaths(slug)
    pp.ensure()
    return pp


def create_project(nome: str, classes_iniciais=None, task: str = 'detect') -> ProjectPaths:
    slug = slugify_projeto(nome)
    if not slug:
        raise ValueError('Nome do modelo nao pode ser vazio.')
    if task not in ('detect', 'segment'):
        raise ValueError('Tipo de treinamento invalido.')
    pp = ProjectPaths(slug)
    if pp.root.exists():
        raise ValueError('Ja existe um modelo com esse nome.')
    pp.ensure()
    save_project_task(pp, task)
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


def build_rtsp_url(ip: str, usuario: str, senha: str, porta: str, caminho: str) -> str:
    ip = (ip or '').strip()
    usuario = (usuario or '').strip()
    senha = (senha or '').strip()
    if not ip or not usuario or not senha:
        raise ValueError('Preencha ip, usuario e senha da camera.')
    porta = (porta or '554').strip() or '554'
    caminho = (caminho or '/cam/realmonitor?channel=1&subtype=0').strip()
    if not caminho.startswith('/'):
        caminho = '/' + caminho
    usuario_escapado = quote(usuario, safe='%')
    senha_escapada = quote(senha, safe='%')
    return f'rtsp://{usuario_escapado}:{senha_escapada}@{ip}:{porta}{caminho}'


def update_config_section(config_path: Path, section: str, updates: dict) -> None:
    """Atualiza (ou adiciona) chaves de uma secao do config.cfg preservando
    comentarios, espacamento e o restante do arquivo como estao."""
    lines = config_path.read_text(encoding='utf-8-sig').splitlines(keepends=True)
    pending = dict(updates)

    section_header = f'[{section}]'
    section_start = None
    section_end = len(lines)
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == section_header:
            section_start = idx
            continue
        if section_start is not None and stripped.startswith('[') and stripped.endswith(']'):
            section_end = idx
            break

    if section_start is None:
        if lines and not lines[-1].endswith('\n'):
            lines[-1] += '\n'
        lines.append(f'\n[{section}]\n')
        lines.extend(f'{key} = {value}\n' for key, value in pending.items())
        config_path.write_text(''.join(lines), encoding='utf-8')
        return

    for idx in range(section_start + 1, section_end):
        stripped = lines[idx].strip()
        if not stripped or stripped.startswith('#') or '=' not in stripped:
            continue
        key = stripped.split('=', 1)[0].strip()
        if key in pending:
            value = pending.pop(key)
            newline = '\n' if lines[idx].endswith('\n') else ''
            lines[idx] = f'{key} = {value}{newline}'

    if pending:
        novas_linhas = [f'{key} = {value}\n' for key, value in pending.items()]
        lines[section_end:section_end] = novas_linhas

    config_path.write_text(''.join(lines), encoding='utf-8')


def remove_config_section(config_path: Path, section: str) -> None:
    """Remove uma secao inteira do config.cfg (usado ao excluir uma camera)."""
    lines = config_path.read_text(encoding='utf-8-sig').splitlines(keepends=True)
    section_header = f'[{section}]'
    section_start = None
    section_end = len(lines)
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == section_header:
            section_start = idx
            continue
        if section_start is not None and stripped.startswith('[') and stripped.endswith(']'):
            section_end = idx
            break
    if section_start is None:
        return
    del lines[section_start:section_end]
    config_path.write_text(''.join(lines), encoding='utf-8')


def load_ids_list(parser: configparser.ConfigParser, section: str, key: str = 'ids') -> list:
    raw = parser.get(section, key, fallback='').strip()
    return [parte.strip() for parte in raw.split(',') if parte.strip()]


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


class CameraManager:
    """Mantem a lista de cameras cadastradas (persistida no config.cfg, em
    [cameras] + [camera_<id>]) e o CameraStream da camera ATIVA no momento,
    usado pela tela de Treinamento (video ao vivo / captura de frames).
    Trocar de camera ativa troca o stream em tempo real, sem reiniciar o
    programa."""

    def __init__(self, parser: configparser.ConfigParser) -> None:
        self.lock = threading.Lock()
        self.cameras = {}
        self.active_id = None
        self.stream = None
        self._proximo_id = parser.getint('cameras', 'proximo_id', fallback=1)
        self._carregar(parser)

    def _carregar(self, parser: configparser.ConfigParser) -> None:
        ids = load_ids_list(parser, 'cameras')
        if not ids and 'camera' in parser:
            # Compatibilidade: config.cfg antigos so tinham uma [camera]
            # unica. Na primeira vez que abrir com o CameraManager, isso
            # migra automaticamente pra virar a "camera 1" da lista nova,
            # sem precisar recadastrar nada.
            cam = parser['camera']
            dados = {
                'nome': 'Camera principal',
                'ip': cam.get('ip', '').strip(),
                'usuario': cam.get('usuario', '').strip(),
                'senha': cam.get('senha', '').strip(),
                'porta': cam.get('porta', '554').strip() or '554',
                'caminho': cam.get('caminho', '/cam/realmonitor?channel=1&subtype=0').strip(),
            }
            self.cameras['1'] = dados
            self._proximo_id = 2
            self.active_id = '1'
            try:
                update_config_section(CONFIG_FILE, 'cameras', {'ids': '1', 'ativa_id': '1', 'proximo_id': '2'})
                update_config_section(CONFIG_FILE, 'camera_1', dados)
            except Exception:
                pass
            return

        for cam_id in ids:
            section = f'camera_{cam_id}'
            self.cameras[cam_id] = {
                'nome': parser.get(section, 'nome', fallback=f'Camera {cam_id}').strip() or f'Camera {cam_id}',
                'ip': parser.get(section, 'ip', fallback='').strip(),
                'usuario': parser.get(section, 'usuario', fallback='').strip(),
                'senha': parser.get(section, 'senha', fallback='').strip(),
                'porta': parser.get(section, 'porta', fallback='554').strip() or '554',
                'caminho': parser.get(section, 'caminho', fallback='/cam/realmonitor?channel=1&subtype=0').strip(),
            }
        ativa = parser.get('cameras', 'ativa_id', fallback='').strip()
        self.active_id = ativa if ativa in self.cameras else (ids[0] if ids else None)

    def iniciar(self) -> None:
        """Conecta na camera ativa (chamado uma vez, ao subir o programa).
        Se falhar, so registra o erro -- o resto do app continua de pe pra
        dar pra trocar/cadastrar outra camera pela tela."""
        if not self.active_id or self.active_id not in self.cameras:
            raise RuntimeError('Nenhuma camera cadastrada ainda. Adicione uma na tela de Cameras.')
        cam = self.cameras[self.active_id]
        rtsp = build_rtsp_url(cam['ip'], cam['usuario'], cam['senha'], cam['porta'], cam['caminho'])
        stream = CameraStream(rtsp)
        if not stream.is_opened():
            stream.stop()
            raise RuntimeError(f"Nao foi possivel abrir o stream RTSP da camera \"{cam['nome']}\". Verifique IP/usuario/senha.")
        stream.start()
        with self.lock:
            self.stream = stream

    def get_frame(self):
        with self.lock:
            stream = self.stream
        return stream.get_frame() if stream is not None else None

    def status(self) -> dict:
        with self.lock:
            conectada = self.stream is not None and self.stream.is_opened()
        return {'ativa_id': self.active_id, 'conectada': conectada}

    def listar(self) -> list:
        """Lista as cameras sem expor a senha (nao precisa ir pro navegador
        so pra mostrar a lista)."""
        return [
            {
                'id': cam_id,
                'ativa': cam_id == self.active_id,
                'nome': dados['nome'],
                'ip': dados['ip'],
                'porta': dados['porta'],
                'caminho': dados['caminho'],
            }
            for cam_id, dados in self.cameras.items()
        ]

    def adicionar(self, nome: str, ip: str, usuario: str, senha: str, porta: str, caminho: str) -> str:
        ip = (ip or '').strip()
        usuario = (usuario or '').strip()
        senha = (senha or '').strip()
        if not ip or not usuario or not senha:
            raise ValueError('Preencha IP, usuario e senha da camera.')
        cam_id = str(self._proximo_id)
        nome = (nome or '').strip() or f'Camera {cam_id}'
        porta = (porta or '554').strip() or '554'
        caminho = (caminho or '/cam/realmonitor?channel=1&subtype=0').strip()
        if not caminho.startswith('/'):
            caminho = '/' + caminho

        self._proximo_id += 1
        dados = {'nome': nome, 'ip': ip, 'usuario': usuario, 'senha': senha, 'porta': porta, 'caminho': caminho}
        self.cameras[cam_id] = dados

        try:
            update_config_section(CONFIG_FILE, 'cameras', {
                'ids': ','.join(self.cameras.keys()),
                'proximo_id': str(self._proximo_id),
            })
            update_config_section(CONFIG_FILE, f'camera_{cam_id}', dados)
        except Exception:
            pass
        return cam_id

    def remover(self, cam_id: str) -> None:
        if cam_id not in self.cameras:
            raise ValueError('Camera nao encontrada.')
        if cam_id == self.active_id:
            raise ValueError('Essa e a camera ativa no momento -- ative outra antes de remover essa.')
        del self.cameras[cam_id]
        try:
            update_config_section(CONFIG_FILE, 'cameras', {'ids': ','.join(self.cameras.keys())})
            remove_config_section(CONFIG_FILE, f'camera_{cam_id}')
        except Exception:
            pass

    def ativar(self, cam_id: str) -> None:
        if cam_id not in self.cameras:
            raise ValueError('Camera nao encontrada.')
        cam = self.cameras[cam_id]
        rtsp = build_rtsp_url(cam['ip'], cam['usuario'], cam['senha'], cam['porta'], cam['caminho'])
        novo_stream = CameraStream(rtsp)
        if not novo_stream.is_opened():
            novo_stream.stop()
            raise RuntimeError('Nao foi possivel conectar nessa camera. Confira IP, usuario, senha e porta.')
        novo_stream.start()
        with self.lock:
            antiga = self.stream
            self.stream = novo_stream
            self.active_id = cam_id
        if antiga is not None:
            antiga.stop()
        try:
            update_config_section(CONFIG_FILE, 'cameras', {'ativa_id': cam_id})
        except Exception:
            pass

    def stop(self) -> None:
        with self.lock:
            stream = self.stream
            self.stream = None
        if stream is not None:
            stream.stop()


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


def hex_to_bgr(hex_color: str):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


_modelo_cache = {}
_modelo_cache_lock = threading.Lock()


def carregar_modelo_cache(path: Path) -> YOLO:
    chave = str(path.resolve())
    with _modelo_cache_lock:
        modelo = _modelo_cache.get(chave)
        if modelo is None:
            modelo = YOLO(chave)
            _modelo_cache[chave] = modelo
        return modelo


def desenhar_deteccoes(frame, resultado) -> list:
    deteccoes = []
    nomes = resultado.names
    masks_xy = resultado.masks.xy if resultado.masks is not None else None
    for i, box in enumerate(resultado.boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        cor_hex = PALETTE[cls_id % len(PALETTE)]
        cor_bgr = hex_to_bgr(cor_hex)
        nome_classe = nomes.get(cls_id, str(cls_id))
        if masks_xy is not None and i < len(masks_xy):
            pontos = masks_xy[i].astype(np.int32)
            cv2.polylines(frame, [pontos], True, cor_bgr, 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), cor_bgr, 2)
        label = f'{nome_classe} {conf:.0%}'
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - lh - 8)), (x1 + lw + 6, y1), cor_bgr, -1)
        cv2.putText(frame, label, (x1 + 3, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (11, 18, 32), 1, cv2.LINE_AA)
        deteccoes.append({'classe': nome_classe, 'confianca': round(conf, 4), 'cor': cor_hex})
    return deteccoes


class TestVideoState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stream = None
        self.running = False
        self.projeto = None
        self.run = None
        self.erro = None
        self.last_jpeg = None

    def start(self, stream, projeto: str, run: str) -> None:
        with self.lock:
            self.stream = stream
            self.projeto = projeto
            self.run = run
            self.running = True
            self.erro = None
            self.last_jpeg = None

    def set_frame(self, jpeg_bytes: bytes) -> None:
        with self.lock:
            self.last_jpeg = jpeg_bytes

    def get_frame(self):
        with self.lock:
            return self.last_jpeg

    def set_erro(self, msg: str) -> None:
        with self.lock:
            self.erro = msg
            self.running = False

    def stop(self) -> None:
        with self.lock:
            stream = self.stream
            self.stream = None
            self.running = False
            self.last_jpeg = None
        if stream is not None:
            stream.stop()

    def snapshot(self) -> dict:
        with self.lock:
            return {'running': self.running, 'projeto': self.projeto, 'run': self.run, 'erro': self.erro}


test_video_state = TestVideoState()


def _loop_teste_video(stream: CameraStream, modelo: YOLO, conf: float) -> None:
    try:
        while test_video_state.running:
            frame = stream.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            resultado = modelo.predict(frame, conf=conf, verbose=False)[0]
            desenhar_deteccoes(frame, resultado)
            ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ok:
                test_video_state.set_frame(buf.tobytes())
            time.sleep(0.03)
    except Exception as exc:
        test_video_state.set_erro(str(exc))
    finally:
        stream.stop()


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

    task = load_project_task(pp)
    if task == 'segment':
        permitidos = {'yolov8n-seg.pt', 'yolov8s-seg.pt', 'yolov8m-seg.pt'}
    else:
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
      --bg: #f5f5f7; --bg-elevated: #ffffff; --card: #ffffff; --border: #d2d2d7;
      --fill: rgba(0,0,0,.045); --fill-strong: rgba(0,0,0,.07);
      --text: #1d1d1f; --text-muted: #6e6e73; --text-faint: #86868b;
      --accent: #0071e3; --accent-tint: rgba(0,113,227,.1); --accent-hover: #0077ed; --accent-press: #005fc0;
      --ok: #22c55e; --err: #ef4444; --warn: #f59e0b; --neutral: #86868b;
      --radius: 20px; --radius-md: 12px; --radius-sm: 8px;
      --shadow-sm: 0 1px 2px rgba(0,0,0,.04), 0 1px 1px rgba(0,0,0,.03);
      --shadow: 0 1px 2px rgba(0,0,0,.03), 0 10px 28px rgba(0,0,0,.06);
      --shadow-lg: 0 24px 64px rgba(0,0,0,.16), 0 4px 14px rgba(0,0,0,.08);
      --ease: cubic-bezier(.25,.1,.25,1);
    }
    * { box-sizing: border-box; }
    html { -webkit-tap-highlight-color: transparent; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Helvetica Neue", Arial, sans-serif;
      background: var(--bg); color: var(--text); font-size: 14px; line-height: 1.47;
      -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale;
      letter-spacing: -.006em;
    }
    button, input, select { font-family: inherit; letter-spacing: inherit; }
    ::selection { background: var(--accent); color: #fff; }

    .app-shell { display: flex; align-items: stretch; min-height: 100vh; }
    .sidebar {
      flex: 0 0 220px; width: 220px; background: var(--bg-elevated); border-right: 1px solid var(--border);
      display: flex; flex-direction: column; gap: 2px; padding: 22px 12px;
    }
    .side-btn {
      display: flex; align-items: center; gap: 10px;
      background: none; border: none; color: var(--text-muted);
      border-radius: 10px; padding: 9px 12px; text-align: left; cursor: pointer;
      font-size: 13.5px; font-weight: 590; letter-spacing: -.006em;
      transition: background-color .15s var(--ease), color .15s var(--ease);
    }
    .side-btn svg { flex: 0 0 auto; width: 18px; height: 18px; opacity: .82; }
    .side-btn:hover { color: var(--text); background: var(--fill); }
    .side-btn.active { color: #fff; background: var(--accent); box-shadow: 0 1px 3px rgba(0,113,227,.35); }
    .side-btn.active svg { opacity: 1; }
    .side-btn:active { transform: scale(.98); }
    .app-main { flex: 1; min-width: 0; }

    header.topbar {
      display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 16px 28px;
      background: rgba(255,255,255,.78); border-bottom: 1px solid var(--border); flex-wrap: wrap;
      position: sticky; top: 0; z-index: 20;
      -webkit-backdrop-filter: saturate(180%) blur(20px); backdrop-filter: saturate(180%) blur(20px);
    }
    header.topbar .brand { display: flex; align-items: baseline; gap: 12px; }
    header.topbar h1 { font-size: 21px; font-weight: 650; letter-spacing: -.02em; margin: 0; }
    header.topbar span { font-size: 12.5px; color: var(--text-muted); }

    .projeto-row { display: flex; align-items: center; gap: 10px; }
    .projeto-row select {
      background: var(--fill); border: 1px solid transparent; color: var(--text);
      border-radius: var(--radius-sm); padding: 8px 12px; font-size: 13px; min-width: 220px;
      transition: background-color .15s var(--ease), border-color .15s var(--ease);
    }
    .projeto-row select:hover { background: var(--fill-strong); }
    .projeto-row select:focus { outline: none; background: #fff; border-color: var(--accent); box-shadow: 0 0 0 4px var(--accent-tint); }

    main.content { max-width: 1360px; margin: 0 auto; padding: 28px 28px 72px; display: flex; flex-direction: column; gap: 24px; }
    .hr-soft { border: none; border-top: 1px solid var(--border); margin: 14px 0; width: 100%; }
    .tester-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    @media (max-width: 720px) { .tester-grid { grid-template-columns: 1fr; } .sidebar { flex-basis: 72px; width: 72px; } .side-btn span.label-text { display: none; } }
    .tester-grid img { max-width: 100%; border-radius: var(--radius-md); background: #000; display: block; }
    .card {
      background: var(--card); border: 1px solid rgba(0,0,0,.04); border-radius: var(--radius);
      padding: 26px; box-shadow: var(--shadow);
    }
    .card h2 {
      margin: 0 0 18px 0; font-size: 18px; font-weight: 600; letter-spacing: -.014em;
      color: var(--text);
    }
    .card-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 18px; }
    .card-header h2 { margin: 0; }

    .video-row { display: flex; gap: 22px; flex-wrap: wrap; align-items: flex-start; }
    .video-frame {
      flex: 1 1 640px; min-width: 320px; max-width: 100%; border-radius: var(--radius-md);
      overflow: hidden; background: #000; box-shadow: var(--shadow-sm); line-height: 0;
    }
    .video-frame img { width: 100%; max-width: 100%; display: block; transition: width .12s linear; }
    .capture-panel { display: flex; flex-direction: column; gap: 18px; flex: 0 0 240px; min-width: 220px; }
    .panel-block { display: flex; flex-direction: column; gap: 9px; }
    .panel-label { font-size: 12px; font-weight: 600; color: var(--text-muted); }
    .panel-divider { height: 1px; background: var(--border); }
    .btn-block { width: 100%; text-align: center; }

    .size-slider-row { display: flex; align-items: center; gap: 10px; }
    .size-slider-row .hint { min-width: 48px; text-align: right; font-variant-numeric: tabular-nums; }
    input[type="range"] {
      -webkit-appearance: none; appearance: none; flex: 1; height: 4px; border-radius: 999px;
      background: var(--fill-strong); outline: none; cursor: pointer;
    }
    input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none; width: 18px; height: 18px; border-radius: 50%; margin-top: -7px;
      background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,.25), 0 0 0 1px rgba(0,0,0,.06); cursor: pointer;
    }
    input[type="range"]::-moz-range-track { height: 4px; border-radius: 999px; background: transparent; }
    input[type="range"]::-moz-range-thumb {
      width: 18px; height: 18px; border-radius: 50%; border: none;
      background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,.25), 0 0 0 1px rgba(0,0,0,.06); cursor: pointer;
    }

    .file-picker {
      display: flex; align-items: center; justify-content: center; gap: 8px;
      background: var(--fill); border: 1.5px dashed var(--border); border-radius: var(--radius-sm);
      padding: 13px 10px; cursor: pointer; font-size: 12.5px; color: var(--text-muted); font-weight: 500;
      transition: background-color .15s var(--ease), border-color .15s var(--ease), color .15s var(--ease);
    }
    .file-picker:hover { background: var(--fill-strong); border-color: var(--accent); color: var(--text); }
    .file-picker svg { flex: 0 0 auto; width: 16px; height: 16px; }
    .hint { font-size: 12.5px; color: var(--text-muted); }
    .hint.ok { color: var(--ok); }
    .hint.err { color: var(--err); }

    .btn-primary {
      background: var(--accent); border: none; border-radius: 980px; padding: 10px 20px;
      color: #fff; font-size: 13.5px; font-weight: 590; cursor: pointer;
      transition: background-color .15s var(--ease), transform .1s var(--ease);
    }
    .btn-primary:hover { background: var(--accent-hover); }
    .btn-primary:active { background: var(--accent-press); transform: scale(.97); }
    .btn-primary:disabled { background: var(--neutral); cursor: not-allowed; transform: none; }
    .btn-secondary {
      background: var(--fill); border: none; border-radius: 980px; padding: 9px 18px;
      color: var(--text); font-size: 13.5px; font-weight: 590; cursor: pointer;
      transition: background-color .15s var(--ease), transform .1s var(--ease);
    }
    .btn-secondary:hover { background: var(--fill-strong); }
    .btn-secondary:active { transform: scale(.97); }

    .chip-list { display: flex; flex-wrap: wrap; gap: 8px; }
    .chip-select {
      border: 1px solid transparent; background: var(--fill); color: var(--text);
      border-radius: 999px; padding: 6px 14px; font-size: 12.5px; font-weight: 500; cursor: pointer;
      border-left: 4px solid var(--chip-color, var(--accent));
      transition: background-color .15s var(--ease), transform .1s var(--ease);
    }
    .chip-select:hover { background: var(--fill-strong); }
    .chip-select:active { transform: scale(.96); }
    .chip-select.active { background: var(--accent-tint); border-color: var(--accent); color: var(--accent); font-weight: 600; }

    .inline-form { display: flex; gap: 8px; margin-top: 14px; }
    .inline-form input {
      flex: 1; max-width: 260px; background: var(--fill); border: 1px solid transparent;
      border-radius: var(--radius-sm); padding: 9px 13px; color: var(--text); font-size: 13.5px;
      transition: background-color .15s var(--ease), border-color .15s var(--ease);
    }
    .inline-form input:focus { outline: none; background: #fff; border-color: var(--accent); box-shadow: 0 0 0 4px var(--accent-tint); }
    .inline-form button {
      background: var(--fill); border: none; border-radius: var(--radius-sm);
      padding: 9px 16px; color: var(--text); font-size: 13.5px; font-weight: 590; cursor: pointer;
      transition: background-color .15s var(--ease), transform .1s var(--ease);
    }
    .inline-form button:hover { background: var(--fill-strong); }
    .inline-form button:active { transform: scale(.97); }

    .tabs { display: inline-flex; gap: 2px; background: var(--fill); border-radius: 10px; padding: 3px; }
    .tab-btn {
      background: transparent; border: 1px solid transparent; color: var(--text-muted);
      border-radius: 8px; padding: 6px 14px; font-size: 12.5px; font-weight: 590; cursor: pointer;
      transition: background-color .18s var(--ease), color .18s var(--ease), box-shadow .18s var(--ease);
    }
    .tab-btn.active { background: #fff; color: var(--text); border-color: transparent; box-shadow: 0 1px 4px rgba(0,0,0,.12); }

    .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 14px; }
    .thumb {
      background: var(--fill); border: 1px solid rgba(0,0,0,.04); border-radius: var(--radius-md);
      overflow: hidden; position: relative; box-shadow: var(--shadow-sm);
      transition: transform .18s var(--ease), box-shadow .18s var(--ease);
    }
    .thumb:hover { transform: translateY(-3px); box-shadow: 0 10px 24px rgba(0,0,0,.12); }
    .thumb img { width: 100%; height: 110px; object-fit: cover; display: block; background: #000; }
    .thumb-badge {
      position: absolute; top: 8px; right: 8px; font-size: 10px; font-weight: 700;
      padding: 3px 8px; border-radius: 999px; background: rgba(0,0,0,.55);
      -webkit-backdrop-filter: blur(6px); backdrop-filter: blur(6px); color: #fff;
    }
    .thumb-badge.ok { color: #6bf29a; }
    .thumb-badge.warn { color: #ffc257; }
    .thumb-classes {
      position: absolute; bottom: 8px; left: 8px; display: flex; gap: 4px;
      padding: 4px 6px; border-radius: 999px; background: rgba(0,0,0,.55);
      -webkit-backdrop-filter: blur(6px); backdrop-filter: blur(6px);
    }
    .thumb-dot { width: 8px; height: 8px; border-radius: 50%; box-shadow: 0 0 0 1px rgba(255,255,255,.5); }
    .thumb-actions { display: flex; flex-direction: column; gap: 5px; padding: 9px; }
    .thumb-btn {
      background: #fff; border: 1px solid rgba(0,0,0,.06); color: var(--text);
      border-radius: var(--radius-sm); padding: 6px 8px; font-size: 11px; font-weight: 500; cursor: pointer;
      transition: background-color .15s var(--ease), color .15s var(--ease), border-color .15s var(--ease);
    }
    .thumb-btn:hover { background: var(--accent-tint); border-color: var(--accent); color: var(--accent); }
    .thumb-btn.danger:hover { background: rgba(239,68,68,.1); border-color: var(--err); color: var(--err); }

    .training-form { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-end; }
    .training-form label {
      display: flex; flex-direction: column; gap: 6px; font-size: 12px; font-weight: 500; color: var(--text-muted);
    }
    .training-form input, .training-form select {
      background: var(--fill); border: 1px solid transparent; border-radius: var(--radius-sm);
      padding: 9px 11px; color: var(--text); font-size: 13.5px; min-width: 140px;
      transition: background-color .15s var(--ease), border-color .15s var(--ease);
    }
    .training-form input:focus, .training-form select:focus {
      outline: none; background: #fff; border-color: var(--accent); box-shadow: 0 0 0 4px var(--accent-tint);
    }

    .training-progress { margin-top: 20px; }
    .progress-bar { background: var(--fill); border-radius: 999px; height: 8px; overflow: hidden; }
    .progress-fill { height: 100%; width: 0%; background: var(--accent); transition: width .4s var(--ease); }
    #progress-text { margin-top: 10px; font-size: 12.5px; color: var(--text-muted); font-weight: 500; }
    .metrics-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
    .metric-pill {
      font-size: 11.5px; background: var(--fill); border-radius: 999px; font-weight: 500;
      padding: 4px 11px; color: var(--text-muted);
    }
    .log-box {
      margin-top: 12px; background: #1d1d1f; border-radius: var(--radius-md);
      padding: 14px; font-size: 11.5px; color: #c8c8cd; max-height: 180px; overflow-y: auto;
      white-space: pre-wrap; font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    }

    .results-list { display: flex; flex-direction: column; gap: 14px; }
    .result-card { background: var(--fill); border-radius: var(--radius-md); padding: 16px; }
    .result-header { display: flex; justify-content: space-between; margin-bottom: 10px; }
    .result-header strong { font-weight: 600; }
    .result-images { display: flex; gap: 8px; overflow-x: auto; margin-bottom: 12px; }
    .result-images img { height: 130px; border-radius: var(--radius-sm); }

    .camera-list { display: flex; flex-direction: column; gap: 10px; margin-top: 16px; }
    .camera-row {
      display: flex; align-items: center; gap: 12px; background: var(--fill);
      border-radius: var(--radius-md); padding: 13px 16px;
      transition: background-color .15s var(--ease);
    }
    .camera-row.ativa { background: var(--accent-tint); box-shadow: inset 0 0 0 1px rgba(0,113,227,.25); }
    .camera-dot {
      width: 9px; height: 9px; border-radius: 50%; background: var(--neutral); flex: 0 0 auto;
      box-shadow: 0 0 0 3px rgba(134,134,139,.15);
    }
    .camera-dot.ok { background: var(--ok); box-shadow: 0 0 0 3px rgba(34,197,94,.18); }
    .camera-dot.err { background: var(--err); box-shadow: 0 0 0 3px rgba(239,68,68,.18); }
    .camera-info { display: flex; flex-direction: column; gap: 2px; flex: 1 1 auto; min-width: 0; }
    .camera-info strong { font-size: 13.5px; font-weight: 600; }
    .camera-info span { font-size: 11.5px; color: var(--text-muted); }
    .camera-badge-ativa {
      font-size: 10.5px; font-weight: 700; letter-spacing: .02em; color: var(--accent);
      background: rgba(0,113,227,.14); border: none;
      border-radius: 999px; padding: 4px 11px; flex: 0 0 auto;
    }
    .camera-actions { display: flex; gap: 8px; flex: 0 0 auto; }
    .camera-actions button { white-space: nowrap; }

    .modal {
      position: fixed; inset: 0; background: rgba(0,0,0,.42); display: none;
      align-items: center; justify-content: center; z-index: 100; padding: 20px;
      -webkit-backdrop-filter: blur(8px); backdrop-filter: blur(8px);
    }
    .modal-box {
      background: var(--card); border: none; border-radius: 24px;
      max-width: 1100px; width: 100%; max-height: 92vh; overflow-y: auto; box-shadow: var(--shadow-lg);
    }
    .modal-header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 18px 22px; border-bottom: 1px solid var(--border);
    }
    .modal-header strong { font-size: 16px; font-weight: 600; letter-spacing: -.012em; }
    .modal-body { display: flex; gap: 20px; padding: 22px; flex-wrap: wrap; }
    .canvas-wrap { flex: 1 1 480px; }
    .canvas-nav-row { display: flex; align-items: center; gap: 10px; }
    .nav-arrow { flex: 0 0 auto; width: 34px; height: 34px; font-size: 18px; border-radius: 999px; }
    .nav-arrow:disabled { opacity: .3; cursor: not-allowed; }
    #anotacao-canvas { flex: 1 1 auto; min-width: 0; max-width: 100%; border-radius: var(--radius-md); background: #000; cursor: crosshair; }
    .modal-side { flex: 0 0 260px; display: flex; flex-direction: column; gap: 6px; }
    .modal-side h3 { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--text-muted); font-weight: 700; margin: 12px 0 6px; }
    .boxes-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 4px; }
    .boxes-list li { font-size: 12.5px; display: flex; align-items: center; gap: 6px; }
    .link-btn-sm { background: none; border: none; color: var(--accent); font-size: 11.5px; font-weight: 500; cursor: pointer; margin-left: auto; }
    .radio-row { display: flex; gap: 14px; font-size: 12.5px; }
    .icon-btn {
      background: var(--fill); border: none; color: var(--text-muted);
      border-radius: var(--radius-sm); width: 30px; height: 30px; cursor: pointer; font-size: 16px; line-height: 1;
      transition: background-color .15s var(--ease), color .15s var(--ease), transform .1s var(--ease);
    }
    .icon-btn:hover { background: var(--fill-strong); color: var(--text); }
    .icon-btn:active { transform: scale(.94); }
    .field-label {
      display: flex; flex-direction: column; gap: 6px; font-size: 12px; font-weight: 500; color: var(--text-muted); width: 100%;
    }
    .field-label input {
      background: var(--fill); border: 1px solid transparent; border-radius: var(--radius-sm);
      padding: 9px 13px; color: var(--text); font-size: 13.5px;
      transition: background-color .15s var(--ease), border-color .15s var(--ease);
    }
    .field-label input:focus { outline: none; background: #fff; border-color: var(--accent); box-shadow: 0 0 0 4px var(--accent-tint); }
  </style>
</head>
<body>
  <div class="app-shell">
    <nav class="sidebar">
      <button type="button" class="side-btn active" data-view="treinamento">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="3"/><path d="M12 3.5v2.2M12 18.3v2.2M20.5 12h-2.2M5.7 12H3.5"/></svg>
        <span class="label-text">Treinamento</span>
      </button>
      <button type="button" class="side-btn" data-view="cameras">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8.5a2 2 0 0 1 2-2h1.2l.9-1.4a2 2 0 0 1 1.68-.9h4.44a2 2 0 0 1 1.68.9l.9 1.4H18a2 2 0 0 1 2 2v8.3a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2Z"/><circle cx="12" cy="13" r="3.4"/></svg>
        <span class="label-text">Cameras</span>
      </button>
      <button type="button" class="side-btn" data-view="testador">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M9 3h6M10 3v5.2L5.6 16a2 2 0 0 0 1.7 3h9.4a2 2 0 0 0 1.7-3L14 8.2V3"/><path d="M7.5 14h9"/></svg>
        <span class="label-text">Testador IA</span>
      </button>
    </nav>
    <div class="app-main">
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

  <main class="content" id="view-treinamento">
    <section class="card">
      <h2>Camera ao vivo</h2>
      <div class="video-row">
        <div class="video-frame">
          <img id="video-feed-img" src="/video_feed" alt="video" />
        </div>
        <div class="capture-panel">
          <div class="panel-block">
            <span class="panel-label">Tamanho do video</span>
            <div class="size-slider-row">
              <input type="range" id="video-tamanho" min="360" max="1180" step="20" value="760" />
              <span id="video-tamanho-valor" class="hint">760px</span>
            </div>
          </div>

          <div class="panel-divider"></div>

          <div class="panel-block">
            <button id="btn-capturar" class="btn-primary btn-block">Capturar frame</button>
            <div id="captura-resultado" class="hint"></div>
          </div>

          <div class="panel-divider"></div>

          <div class="panel-block">
            <span class="panel-label">Ou envie fotos prontas</span>
            <label class="file-picker" for="input-upload-fotos">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V4M12 4 7.5 8.5M12 4l4.5 4.5"/><path d="M4 15.5V18a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2.5"/></svg>
              <span id="upload-picker-texto">Escolher fotos</span>
            </label>
            <input type="file" id="input-upload-fotos" accept="image/jpeg,image/png" multiple style="display:none" />
            <button id="btn-upload-fotos" class="btn-secondary btn-block">Enviar fotos</button>
            <div id="upload-resultado" class="hint"></div>
          </div>
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

  <main class="content" id="view-testador" style="display:none;">
    <section class="card">
      <h2>Testador de IA - Imagem</h2>
      <p class="hint">Escolha um modelo ja treinado (de qualquer um dos seus modelos/projetos), envie uma foto e veja o resultado da deteccao.</p>
      <div class="training-form">
        <label>Modelo treinado
          <select id="test-sel-modelo" style="min-width:280px;"></select>
        </label>
        <label>Confianca minima
          <input type="number" id="test-conf" value="0.4" min="0.05" max="0.95" step="0.05" style="min-width:90px;" />
        </label>
      </div>
      <div class="capture-panel" style="margin-top:14px; max-width:320px;">
        <input type="file" id="test-input-imagem" accept="image/jpeg,image/png" />
        <button id="btn-testar-imagem" class="btn-primary">Testar imagem</button>
        <div id="test-imagem-resultado" class="hint"></div>
      </div>
      <div class="tester-grid" style="margin-top:16px;">
        <div>
          <h3 style="font-size:11px; text-transform:uppercase; color:var(--text-muted); margin:0 0 6px;">Original</h3>
          <img id="test-img-original" style="display:none;" />
        </div>
        <div>
          <h3 style="font-size:11px; text-transform:uppercase; color:var(--text-muted); margin:0 0 6px;">Deteccoes</h3>
          <img id="test-img-resultado" style="display:none;" />
        </div>
      </div>
      <div id="test-deteccoes-lista" class="chip-list" style="margin-top:12px;"></div>
    </section>

    <section class="card">
      <div class="card-header">
        <h2>Testador de IA - Video ao vivo (opcional)</h2>
        <label class="hint" style="display:flex; align-items:center; gap:6px; cursor:pointer;">
          <input type="checkbox" id="test-video-ativo" /> Ativar teste de video
        </label>
      </div>
      <div id="test-video-config" style="display:none;">
        <p class="hint">Informe os dados de conexao de uma camera (pode ser a mesma da captura ou outra qualquer) para testar o modelo ao vivo.</p>
        <div class="training-form">
          <label>IP <input type="text" id="test-cam-ip" placeholder="10.0.0.10" /></label>
          <label>Usuario <input type="text" id="test-cam-usuario" placeholder="admin" /></label>
          <label>Senha <input type="password" id="test-cam-senha" /></label>
          <label>Porta <input type="text" id="test-cam-porta" value="554" style="min-width:70px;" /></label>
          <label>Caminho <input type="text" id="test-cam-caminho" placeholder="/cam/realmonitor?channel=1&amp;subtype=0" style="min-width:260px;" /></label>
          <label>Modelo
            <select id="test-video-sel-modelo" style="min-width:220px;"></select>
          </label>
          <label>Confianca
            <input type="number" id="test-video-conf" value="0.4" min="0.05" max="0.95" step="0.05" style="min-width:90px;" />
          </label>
          <button id="btn-video-conectar" class="btn-primary">Conectar e testar</button>
        </div>
        <div id="test-video-resultado" class="hint" style="margin-top:8px;"></div>
        <div id="test-video-wrap" style="display:none; margin-top:14px;">
          <img id="test-video-img" style="max-width:640px; width:100%; border-radius:10px; background:#000; display:block;" />
          <button id="btn-video-parar" class="btn-secondary" style="margin-top:10px;">Parar teste</button>
        </div>
      </div>
    </section>
  </main>

  <main class="content" id="view-cameras" style="display:none;">
    <section class="card">
      <div class="card-header">
        <h2>Cameras cadastradas</h2>
        <button type="button" id="btn-adicionar-camera" class="btn-primary">+ Adicionar camera</button>
      </div>
      <p class="hint">A camera ativa e usada na tela de Treinamento (video ao vivo e captura de frames). Ativar outra troca na hora, sem precisar reiniciar o programa.</p>
      <div id="lista-cameras" class="camera-list"><p class="hint">Carregando cameras...</p></div>
    </section>
  </main>
    </div>
  </div>

  <div id="modal-anotacao" class="modal">
    <div class="modal-box">
      <div class="modal-header">
        <div style="display:flex; align-items:baseline; gap:10px;">
          <strong id="modal-titulo">Anotar imagem</strong>
          <span id="modal-posicao" class="hint"></span>
        </div>
        <button id="modal-fechar" class="icon-btn">&times;</button>
      </div>
      <div class="modal-body">
        <div class="canvas-wrap">
          <div class="canvas-nav-row">
            <button type="button" id="btn-anotacao-anterior" class="icon-btn nav-arrow" title="Foto anterior">&lsaquo;</button>
            <canvas id="anotacao-canvas"></canvas>
            <button type="button" id="btn-anotacao-proxima" class="icon-btn nav-arrow" title="Proxima foto">&rsaquo;</button>
          </div>
        </div>
        <div class="modal-side">
          <h3>Classe ativa (usada ao desenhar)</h3>
          <div id="modal-classes" class="chip-list"></div>
          <div id="modo-anotacao-row" class="radio-row" style="display:none;">
            <label><input type="radio" name="modo-anotacao" value="rect" checked /> Retangulo</label>
            <label><input type="radio" name="modo-anotacao" value="poligono" /> Poligono</label>
          </div>
          <p id="modo-anotacao-hint" class="hint" style="display:none;">Clique nos vertices do objeto e feche clicando de novo no ponto inicial (min. 3 pontos). Esc cancela.</p>
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
        <label class="field-label" style="margin-top: 10px;">Tipo de treinamento</label>
        <div class="radio-row">
          <label><input type="radio" name="novo-projeto-tipo" value="detect" checked /> Deteccao (caixa)</label>
          <label><input type="radio" name="novo-projeto-tipo" value="segment" /> Segmentacao (contorno preciso)</label>
        </div>
        <button id="btn-criar-projeto" class="btn-primary" style="margin-top: 14px;">Criar modelo</button>
        <div id="novo-projeto-resultado" class="hint"></div>
      </div>
    </div>
  </div>

  <div id="modal-camera" class="modal">
    <div class="modal-box" style="max-width: 440px;">
      <div class="modal-header">
        <strong>Adicionar camera</strong>
        <button id="modal-camera-fechar" class="icon-btn">&times;</button>
      </div>
      <div class="modal-body" style="flex-direction: column;">
        <label class="field-label">Nome (so pra identificar na lista)
          <input type="text" id="camera-nome" placeholder="Ex: Camera Linha 2" maxlength="80" autocomplete="off" />
        </label>
        <label class="field-label" style="margin-top: 10px;">IP
          <input type="text" id="camera-ip" placeholder="10.0.0.10" autocomplete="off" />
        </label>
        <label class="field-label" style="margin-top: 10px;">Usuario
          <input type="text" id="camera-usuario" placeholder="admin" autocomplete="off" />
        </label>
        <label class="field-label" style="margin-top: 10px;">Senha
          <input type="password" id="camera-senha" autocomplete="off" />
        </label>
        <label class="field-label" style="margin-top: 10px;">Porta
          <input type="text" id="camera-porta" value="554" autocomplete="off" />
        </label>
        <label class="field-label" style="margin-top: 10px;">Caminho RTSP
          <input type="text" id="camera-caminho" placeholder="/cam/realmonitor?channel=1&amp;subtype=0" autocomplete="off" />
        </label>
        <button id="btn-salvar-camera" class="btn-primary" style="margin-top: 14px;">Salvar camera</button>
        <div id="camera-resultado" class="hint"></div>
      </div>
    </div>
  </div>

  <script>
    const PALETTE = ['#22c55e', '#f59e0b', '#3b82f6', '#ef4444', '#a855f7', '#14b8a6', '#eab308', '#ec4899'];
    const state = { classes: [], activeClassIndex: 0, currentImage: null, boxes: [], drawing: false, mode: 'rect', polyPoints: [] };
    let abaAtiva = 'pendentes';
    let projetoAtual = localStorage.getItem('treinamento_projeto_atual') || '';
    let projetoTask = 'detect';
    let modoAnotacaoPreferido = localStorage.getItem('treinamento_modo_anotacao') || 'rect';

    document.querySelectorAll('.side-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.side-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const view = btn.getAttribute('data-view');
        document.getElementById('view-treinamento').style.display = view === 'treinamento' ? 'flex' : 'none';
        document.getElementById('view-cameras').style.display = view === 'cameras' ? 'flex' : 'none';
        document.getElementById('view-testador').style.display = view === 'testador' ? 'flex' : 'none';
        if (view === 'testador') carregarModelosTestador();
        if (view === 'cameras') carregarCameras();
      });
    });

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
      document.querySelector('input[name="novo-projeto-tipo"][value="detect"]').checked = true;
      modalNovoProjeto.style.display = 'flex';
    });
    document.getElementById('modal-novo-projeto-fechar').addEventListener('click', () => { modalNovoProjeto.style.display = 'none'; });

    document.getElementById('btn-criar-projeto').addEventListener('click', async () => {
      const nome = document.getElementById('novo-projeto-nome').value.trim();
      const classesRaw = document.getElementById('novo-projeto-classes').value.trim();
      const resultEl = document.getElementById('novo-projeto-resultado');
      if (!nome) { resultEl.textContent = 'Digite um nome.'; resultEl.className = 'hint err'; return; }
      const classes_iniciais = classesRaw ? classesRaw.split(',').map((c) => c.trim()).filter(Boolean) : [];
      const tipo = document.querySelector('input[name="novo-projeto-tipo"]:checked').value;
      const res = await fetch('/projetos', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ nome, classes_iniciais, tipo }),
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

    const modalCamera = document.getElementById('modal-camera');
    document.getElementById('btn-adicionar-camera').addEventListener('click', () => {
      ['camera-nome', 'camera-ip', 'camera-usuario', 'camera-senha', 'camera-caminho'].forEach((id) => {
        document.getElementById(id).value = '';
      });
      document.getElementById('camera-porta').value = '554';
      document.getElementById('camera-resultado').textContent = '';
      modalCamera.style.display = 'flex';
    });
    document.getElementById('modal-camera-fechar').addEventListener('click', () => { modalCamera.style.display = 'none'; });

    document.getElementById('btn-salvar-camera').addEventListener('click', async () => {
      const resultEl = document.getElementById('camera-resultado');
      const payload = {
        nome: document.getElementById('camera-nome').value.trim(),
        ip: document.getElementById('camera-ip').value.trim(),
        usuario: document.getElementById('camera-usuario').value.trim(),
        senha: document.getElementById('camera-senha').value,
        porta: document.getElementById('camera-porta').value.trim(),
        caminho: document.getElementById('camera-caminho').value.trim(),
      };
      if (!payload.ip || !payload.usuario || !payload.senha) {
        resultEl.textContent = 'Preencha pelo menos IP, usuario e senha.';
        resultEl.className = 'hint err';
        return;
      }
      resultEl.textContent = 'Salvando...';
      resultEl.className = 'hint';
      try {
        const res = await fetch('/cameras', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (data.ok) {
          resultEl.textContent = 'Camera adicionada!';
          resultEl.className = 'hint ok';
          setTimeout(async () => { modalCamera.style.display = 'none'; await carregarCameras(); }, 400);
        } else {
          resultEl.textContent = data.erro || 'Falha ao adicionar camera.';
          resultEl.className = 'hint err';
        }
      } catch (_) {
        resultEl.textContent = 'Falha ao contatar o servidor.';
        resultEl.className = 'hint err';
      }
    });

    async function carregarCameras() {
      const wrap = document.getElementById('lista-cameras');
      try {
        const res = await fetch('/cameras');
        const data = await res.json();
        const cameras = data.cameras || [];
        const status = data.status || {};
        if (!cameras.length) {
          wrap.innerHTML = '<p class="hint">Nenhuma camera cadastrada ainda. Clique em "+ Adicionar camera".</p>';
          return;
        }
        wrap.innerHTML = cameras.map((c) => {
          const conectada = c.ativa && status.conectada;
          const dotCls = !c.ativa ? 'camera-dot' : (conectada ? 'camera-dot ok' : 'camera-dot err');
          const statusTexto = !c.ativa ? 'Inativa' : (conectada ? 'Ativa e conectada' : 'Ativa, conectando...');
          return `
            <div class="camera-row ${c.ativa ? 'ativa' : ''}">
              <span class="${dotCls}"></span>
              <div class="camera-info">
                <strong>${c.nome}</strong>
                <span>${c.ip}:${c.porta}${c.caminho} &middot; ${statusTexto}</span>
              </div>
              ${c.ativa ? '<span class="camera-badge-ativa">ATIVA</span>' : ''}
              <div class="camera-actions">
                ${c.ativa ? '' : `<button type="button" class="btn-secondary btn-ativar-camera" data-id="${c.id}">Ativar</button>`}
                ${c.ativa ? '' : `<button type="button" class="btn-secondary btn-remover-camera" data-id="${c.id}">Remover</button>`}
              </div>
            </div>
          `;
        }).join('');
        wrap.querySelectorAll('.btn-ativar-camera').forEach((btn) => {
          btn.addEventListener('click', () => ativarCamera(btn.getAttribute('data-id'), btn));
        });
        wrap.querySelectorAll('.btn-remover-camera').forEach((btn) => {
          btn.addEventListener('click', () => removerCamera(btn.getAttribute('data-id')));
        });
      } catch (_) {
        wrap.innerHTML = '<p class="hint err">Falha ao carregar as cameras.</p>';
      }
    }

    async function ativarCamera(id, btn) {
      const textoOriginal = btn.textContent;
      btn.disabled = true;
      btn.textContent = 'Conectando...';
      try {
        const res = await fetch(`/cameras/${id}/ativar`, { method: 'POST' });
        const data = await res.json();
        if (!data.ok) {
          alert(data.erro || 'Falha ao ativar essa camera.');
          btn.disabled = false;
          btn.textContent = textoOriginal;
          return;
        }
        await carregarCameras();
      } catch (_) {
        alert('Falha ao contatar o servidor.');
        btn.disabled = false;
        btn.textContent = textoOriginal;
      }
    }

    async function removerCamera(id) {
      if (!confirm('Remover essa camera da lista?')) return;
      try {
        const res = await fetch(`/cameras/${id}/remover`, { method: 'POST' });
        const data = await res.json();
        if (!data.ok) {
          alert(data.erro || 'Falha ao remover essa camera.');
          return;
        }
        await carregarCameras();
      } catch (_) {
        alert('Falha ao contatar o servidor.');
      }
    }

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

    (function () {
      const slider = document.getElementById('video-tamanho');
      const valorEl = document.getElementById('video-tamanho-valor');
      const img = document.getElementById('video-feed-img');
      function aplicarTamanhoVideo(v) {
        img.style.width = `${v}px`;
        valorEl.textContent = `${v}px`;
      }
      let salvo = null;
      try { salvo = localStorage.getItem('treinamento_video_largura'); } catch (_) {}
      if (salvo && !Number.isNaN(parseInt(salvo, 10))) slider.value = salvo;
      aplicarTamanhoVideo(slider.value);
      slider.addEventListener('input', () => {
        aplicarTamanhoVideo(slider.value);
        try { localStorage.setItem('treinamento_video_largura', slider.value); } catch (_) {}
      });
    })();

    const inputUploadFotos = document.getElementById('input-upload-fotos');
    const uploadPickerTexto = document.getElementById('upload-picker-texto');
    inputUploadFotos.addEventListener('change', () => {
      const n = inputUploadFotos.files.length;
      uploadPickerTexto.textContent = n === 0 ? 'Escolher fotos' : n === 1 ? '1 foto selecionada' : `${n} fotos selecionadas`;
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

    document.getElementById('btn-upload-fotos').addEventListener('click', async () => {
      if (!projetoAtual) { alert('Crie um modelo primeiro.'); return; }
      const input = document.getElementById('input-upload-fotos');
      const resultEl = document.getElementById('upload-resultado');
      if (!input.files.length) { resultEl.textContent = 'Escolha ao menos uma foto.'; resultEl.className = 'hint err'; return; }
      const fd = new FormData();
      fd.append('projeto', projetoAtual);
      for (const arquivo of input.files) fd.append('arquivos', arquivo);
      resultEl.textContent = 'Enviando...';
      resultEl.className = 'hint';
      try {
        const res = await fetch('/imagens/upload', { method: 'POST', body: fd });
        const data = await res.json();
        if (data.ok) {
          resultEl.textContent = `${data.salvos} foto(s) enviada(s) para "Pendentes".`;
          resultEl.className = 'hint ok';
          input.value = '';
          uploadPickerTexto.textContent = 'Escolher fotos';
          carregarImagens();
        } else {
          resultEl.textContent = data.erro || 'Falha ao enviar.';
          resultEl.className = 'hint err';
        }
      } catch (_) {
        resultEl.textContent = 'Falha ao contatar o servidor.';
        resultEl.className = 'hint err';
      }
    });

    let galeriaItens = [];

    async function carregarImagens() {
      if (!projetoAtual) return;
      const res = await fetch(`/imagens?${qp()}`);
      const data = await res.json();
      document.getElementById('count-pendentes').textContent = data.pendentes.length;
      document.getElementById('count-treino').textContent = data.treino.length;
      document.getElementById('count-validacao').textContent = data.validacao.length;
      projetoTask = data.task || 'detect';
      atualizarSeletorModeloBase();
      galeriaItens = data[abaAtiva] || [];
      renderGaleria(galeriaItens);
    }

    function atualizarSeletorModeloBase() {
      const sel = document.getElementById('cfg-modelo');
      const valorAtual = sel.value;
      const opcoes = projetoTask === 'segment'
        ? [
            ['yolov8n-seg.pt', 'YOLOv8n-seg (rapido, leve, precisa internet)'],
            ['yolov8s-seg.pt', 'YOLOv8s-seg (equilibrado, precisa internet)'],
            ['yolov8m-seg.pt', 'YOLOv8m-seg (mais preciso, precisa internet)'],
          ]
        : [
            ['yolov8n.pt', 'YOLOv8n (rapido, leve)'],
            ['yolov8s.pt', 'YOLOv8s (equilibrado, precisa internet)'],
            ['yolov8m.pt', 'YOLOv8m (mais preciso, precisa internet)'],
          ];
      sel.innerHTML = opcoes.map(([v, label]) => `<option value="${v}">${label}</option>`).join('')
        + '<option value="__ultimo__">Continuar do ultimo treinamento deste modelo</option>';
      if (opcoes.some(([v]) => v === valorAtual) || valorAtual === '__ultimo__') sel.value = valorAtual;
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
          ${item.classes && item.classes.length ? `
            <div class="thumb-classes" title="Classes nesta imagem: ${item.classes.map((i) => state.classes[i] || `classe ${i}`).join(', ')}">
              ${item.classes.map((i) => `<span class="thumb-dot" style="background:${PALETTE[i % PALETTE.length]}"></span>`).join('')}
            </div>
          ` : ''}
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
      state.polyPoints = [];
      document.getElementById('modal-titulo').textContent = `Anotar: ${nome}`;
      document.getElementById('anotacao-resultado').textContent = '';
      const modoRow = document.getElementById('modo-anotacao-row');
      const modoHint = document.getElementById('modo-anotacao-hint');
      const mostrarModo = projetoTask === 'segment';
      modoRow.style.display = mostrarModo ? 'flex' : 'none';
      modoHint.style.display = mostrarModo ? 'block' : 'none';
      state.mode = mostrarModo ? modoAnotacaoPreferido : 'rect';
      document.querySelector(`input[name="modo-anotacao"][value="${state.mode}"]`).checked = true;
      renderClassChips();
      renderListaCaixas();
      atualizarNavegacaoAnotacao();

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
        const maxW = Math.min(760, window.innerWidth - 430);
        const scale = Math.min(1, maxW / img.naturalWidth);
        canvas.width = Math.round(img.naturalWidth * scale);
        canvas.height = Math.round(img.naturalHeight * scale);
        renderListaCaixas();
      };
      img.src = `/imagem?${qp({ pasta, nome })}`;

      modal.style.display = 'flex';
    }

    function atualizarNavegacaoAnotacao() {
      const btnAnt = document.getElementById('btn-anotacao-anterior');
      const btnProx = document.getElementById('btn-anotacao-proxima');
      const posEl = document.getElementById('modal-posicao');
      const idx = galeriaItens.findIndex((it) => it.nome === state.currentImage.nome);
      if (idx === -1) {
        btnAnt.disabled = true;
        btnProx.disabled = true;
        posEl.textContent = '';
        return;
      }
      btnAnt.disabled = idx <= 0;
      btnProx.disabled = idx >= galeriaItens.length - 1;
      posEl.textContent = `${idx + 1} de ${galeriaItens.length}`;
    }

    function navegarAnotacao(direcao) {
      if (!state.currentImage) return;
      const idx = galeriaItens.findIndex((it) => it.nome === state.currentImage.nome);
      if (idx === -1) return;
      const novoIdx = idx + direcao;
      if (novoIdx < 0 || novoIdx >= galeriaItens.length) return;
      abrirAnotacao(state.currentImage.pasta, galeriaItens[novoIdx].nome);
    }

    document.getElementById('btn-anotacao-anterior').addEventListener('click', () => navegarAnotacao(-1));
    document.getElementById('btn-anotacao-proxima').addEventListener('click', () => navegarAnotacao(1));

    document.getElementById('modal-fechar').addEventListener('click', () => { modal.style.display = 'none'; });

    function desenharPoligono(pontosCanvas, cor, fechado) {
      ctx.strokeStyle = cor; ctx.lineWidth = 2;
      ctx.beginPath();
      pontosCanvas.forEach(([x, y], i) => { if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
      if (fechado) ctx.closePath();
      ctx.stroke();
    }

    function desenharCanvas() {
      if (!imagemCarregada) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(imagemCarregada, 0, 0, canvas.width, canvas.height);
      state.boxes.forEach((box) => {
        const cor = PALETTE[box.classe % PALETTE.length];
        let x, y;
        if (box.pontos) {
          const pontosCanvas = box.pontos.map(([px, py]) => [px * canvas.width, py * canvas.height]);
          desenharPoligono(pontosCanvas, cor, true);
          x = Math.min(...pontosCanvas.map((p) => p[0]));
          y = Math.min(...pontosCanvas.map((p) => p[1]));
        } else {
          x = box.x * canvas.width; y = box.y * canvas.height;
          const w = box.w * canvas.width, h = box.h * canvas.height;
          ctx.strokeStyle = cor; ctx.lineWidth = 2;
          ctx.strokeRect(x, y, w, h);
        }
        const nomeClasse = state.classes[box.classe] || '?';
        ctx.font = '11px sans-serif';
        const labelW = ctx.measureText(nomeClasse).width + 8;
        ctx.fillStyle = cor;
        ctx.fillRect(x, Math.max(0, y - 16), labelW, 16);
        ctx.fillStyle = '#0b1220';
        ctx.fillText(nomeClasse, x + 4, Math.max(11, y - 4));
      });
      if (state.mode === 'rect' && state.drawing) {
        ctx.strokeStyle = PALETTE[state.activeClassIndex % PALETTE.length];
        ctx.setLineDash([4, 3]);
        ctx.strokeRect(state.drawX, state.drawY, state.drawW, state.drawH);
        ctx.setLineDash([]);
      }
      if (state.mode === 'poligono' && state.polyPoints.length) {
        const cor = PALETTE[state.activeClassIndex % PALETTE.length];
        desenharPoligono(state.polyPoints, cor, false);
        if (state.mouseX !== undefined) {
          ctx.strokeStyle = cor;
          ctx.setLineDash([4, 3]);
          ctx.beginPath();
          const [lx, ly] = state.polyPoints[state.polyPoints.length - 1];
          ctx.moveTo(lx, ly);
          ctx.lineTo(state.mouseX, state.mouseY);
          ctx.stroke();
          ctx.setLineDash([]);
        }
        const [fx, fy] = state.polyPoints[0];
        ctx.beginPath();
        ctx.arc(fx, fy, 5, 0, Math.PI * 2);
        ctx.fillStyle = cor;
        ctx.fill();
      }
    }

    function posicaoNoCanvas(ev) {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      return [(ev.clientX - rect.left) * scaleX, (ev.clientY - rect.top) * scaleY];
    }

    canvas.addEventListener('mousedown', (ev) => {
      if (state.mode !== 'rect') return;
      if (!state.classes.length) { alert('Cadastre uma classe antes de desenhar.'); return; }
      const [px, py] = posicaoNoCanvas(ev);
      state.drawing = true;
      state.startX = px;
      state.startY = py;
      state.drawX = state.startX; state.drawY = state.startY; state.drawW = 0; state.drawH = 0;
    });

    canvas.addEventListener('mousemove', (ev) => {
      const [curX, curY] = posicaoNoCanvas(ev);
      if (state.mode === 'rect') {
        if (!state.drawing) return;
        state.drawX = Math.min(state.startX, curX);
        state.drawY = Math.min(state.startY, curY);
        state.drawW = Math.abs(curX - state.startX);
        state.drawH = Math.abs(curY - state.startY);
        desenharCanvas();
      } else if (state.mode === 'poligono' && state.polyPoints.length) {
        state.mouseX = curX; state.mouseY = curY;
        desenharCanvas();
      }
    });

    window.addEventListener('mouseup', () => {
      if (state.mode !== 'rect' || !state.drawing) return;
      state.drawing = false;
      if (state.drawW > 4 && state.drawH > 4 && canvas.width > 0) {
        const x0 = state.drawX / canvas.width, y0 = state.drawY / canvas.height;
        const w0 = state.drawW / canvas.width, h0 = state.drawH / canvas.height;
        if (projetoTask === 'segment') {
          state.boxes.push({
            classe: state.activeClassIndex,
            pontos: [[x0, y0], [x0 + w0, y0], [x0 + w0, y0 + h0], [x0, y0 + h0]],
          });
        } else {
          state.boxes.push({ classe: state.activeClassIndex, x: x0, y: y0, w: w0, h: h0 });
        }
        renderListaCaixas();
      } else {
        desenharCanvas();
      }
    });

    canvas.addEventListener('click', (ev) => {
      if (state.mode !== 'poligono') return;
      if (!state.classes.length) { alert('Cadastre uma classe antes de desenhar.'); return; }
      const [x, y] = posicaoNoCanvas(ev);
      if (state.polyPoints.length >= 3) {
        const [fx, fy] = state.polyPoints[0];
        if (Math.hypot(x - fx, y - fy) <= 8) {
          const pontos = state.polyPoints.map(([px, py]) => [px / canvas.width, py / canvas.height]);
          state.boxes.push({ classe: state.activeClassIndex, pontos });
          state.polyPoints = [];
          state.mouseX = undefined;
          renderListaCaixas();
          return;
        }
      }
      state.polyPoints.push([x, y]);
      desenharCanvas();
    });

    document.querySelectorAll('input[name="modo-anotacao"]').forEach((r) => {
      r.addEventListener('change', (ev) => {
        state.mode = ev.target.value;
        modoAnotacaoPreferido = state.mode;
        localStorage.setItem('treinamento_modo_anotacao', modoAnotacaoPreferido);
        state.polyPoints = [];
        state.drawing = false;
        desenharCanvas();
      });
    });

    window.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape' && state.mode === 'poligono' && state.polyPoints.length) {
        state.polyPoints = [];
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
      const pastaOrigem = state.currentImage.pasta;
      const caixas = projetoTask === 'segment'
        ? state.boxes.map((b) => ({ classe: b.classe, pontos: b.pontos }))
        : state.boxes.map((b) => ({ classe: b.classe, x: b.x + b.w / 2, y: b.y + b.h / 2, w: b.w, h: b.h }));
      const res = await fetch('/anotar', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          projeto: projetoAtual,
          pasta: pastaOrigem,
          nome: state.currentImage.nome,
          destino,
          caixas,
        }),
      });
      const data = await res.json();
      if (data.ok) {
        resultEl.textContent = 'Salvo!';
        resultEl.className = 'hint ok';
        // Se a imagem ja estava organizada (treino/validacao) e voce esta so
        // passando de uma pra outra revisando, nao faz sentido fechar o modal
        // a cada save -- perderia o lugar de onde estava. Mantem a imagem
        // aberta, so atualiza a galeria (contadores/miniaturas) por tras.
        // Pra quem esta triando capturas novas (pendentes), continua fechando
        // e voltando pra lista, que e o fluxo que ja fazia sentido ali.
        if (pastaOrigem === 'treino' || pastaOrigem === 'validacao') {
          state.currentImage.pasta = destino;
          await carregarImagens();
          atualizarNavegacaoAnotacao();
          setTimeout(() => { resultEl.textContent = ''; resultEl.className = 'hint'; }, 1500);
        } else {
          setTimeout(() => { modal.style.display = 'none'; carregarImagens(); }, 500);
        }
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
          ${r.tem_modelo ? `
            <div style="display:flex; gap:8px; flex-wrap:wrap;">
              <button class="btn-primary" data-run="${r.nome}">Usar este modelo</button>
              <a class="btn-secondary" style="text-decoration:none; display:inline-flex; align-items:center;" href="/treinamento/resultados/${r.nome}/modelo/baixar?${qp()}">Baixar modelo (.pt)</a>
            </div>
          ` : '<span class="hint">Sem best.pt gerado</span>'}
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

    async function carregarModelosTestador() {
      const res = await fetch('/testador/modelos');
      const lista = await res.json();
      const optionsHtml = lista.length
        ? lista.map((m) => `<option value="${m.projeto}|${m.run}">${m.projeto} - ${m.run} (${m.modificado})</option>`).join('')
        : '<option value="">Nenhum modelo treinado ainda</option>';
      document.getElementById('test-sel-modelo').innerHTML = optionsHtml;
      document.getElementById('test-video-sel-modelo').innerHTML = optionsHtml;
    }

    document.getElementById('btn-testar-imagem').addEventListener('click', async () => {
      const sel = document.getElementById('test-sel-modelo').value;
      const resultEl = document.getElementById('test-imagem-resultado');
      const input = document.getElementById('test-input-imagem');
      if (!sel) { resultEl.textContent = 'Nenhum modelo treinado disponivel.'; resultEl.className = 'hint err'; return; }
      if (!input.files.length) { resultEl.textContent = 'Escolha uma imagem.'; resultEl.className = 'hint err'; return; }
      const [projeto, run] = sel.split('|');

      const origPreview = document.getElementById('test-img-original');
      origPreview.src = URL.createObjectURL(input.files[0]);
      origPreview.style.display = 'block';

      const fd = new FormData();
      fd.append('projeto', projeto);
      fd.append('run', run);
      fd.append('conf', document.getElementById('test-conf').value);
      fd.append('imagem', input.files[0]);

      resultEl.textContent = 'Analisando...';
      resultEl.className = 'hint';
      try {
        const res = await fetch('/testador/imagem', { method: 'POST', body: fd });
        const data = await res.json();
        if (data.ok) {
          const resImg = document.getElementById('test-img-resultado');
          resImg.src = `data:image/jpeg;base64,${data.imagem}`;
          resImg.style.display = 'block';
          resultEl.textContent = data.deteccoes.length ? `${data.deteccoes.length} deteccao(oes) encontrada(s).` : 'Nenhuma deteccao encontrada.';
          resultEl.className = 'hint ok';
          document.getElementById('test-deteccoes-lista').innerHTML = data.deteccoes.map((d) => `
            <span class="chip-select" style="--chip-color:${d.cor}">${d.classe} (${Math.round(d.confianca * 100)}%)</span>
          `).join('');
        } else {
          resultEl.textContent = data.erro || 'Falha ao testar imagem.';
          resultEl.className = 'hint err';
        }
      } catch (_) {
        resultEl.textContent = 'Falha ao contatar o servidor.';
        resultEl.className = 'hint err';
      }
    });

    let configCameraCarregada = false;
    async function preencherConfigCameraPadrao() {
      if (configCameraCarregada) return;
      configCameraCarregada = true;
      try {
        const res = await fetch('/config/camera');
        const cfg = await res.json();
        const campos = {
          'test-cam-ip': cfg.ip, 'test-cam-usuario': cfg.usuario, 'test-cam-senha': cfg.senha,
          'test-cam-porta': cfg.porta, 'test-cam-caminho': cfg.caminho,
        };
        Object.entries(campos).forEach(([id, valor]) => {
          const el = document.getElementById(id);
          if (el && valor) el.value = valor;
        });
      } catch (_) {}
    }

    document.getElementById('test-video-ativo').addEventListener('change', (ev) => {
      document.getElementById('test-video-config').style.display = ev.target.checked ? 'block' : 'none';
      if (ev.target.checked) preencherConfigCameraPadrao();
    });

    document.getElementById('btn-video-conectar').addEventListener('click', async () => {
      const sel = document.getElementById('test-video-sel-modelo').value;
      const resultEl = document.getElementById('test-video-resultado');
      if (!sel) { resultEl.textContent = 'Nenhum modelo treinado disponivel.'; resultEl.className = 'hint err'; return; }
      const [projeto, run] = sel.split('|');
      const body = {
        ip: document.getElementById('test-cam-ip').value,
        usuario: document.getElementById('test-cam-usuario').value,
        senha: document.getElementById('test-cam-senha').value,
        porta: document.getElementById('test-cam-porta').value,
        caminho: document.getElementById('test-cam-caminho').value,
        projeto, run,
        conf: document.getElementById('test-video-conf').value,
      };
      const btn = document.getElementById('btn-video-conectar');
      btn.disabled = true;
      resultEl.textContent = 'Conectando na camera...';
      resultEl.className = 'hint';
      try {
        const res = await fetch('/testador/video/iniciar', {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
        });
        const data = await res.json();
        if (data.ok) {
          resultEl.textContent = 'Conectado! Exibindo video com as deteccoes ao vivo.';
          resultEl.className = 'hint ok';
          document.getElementById('test-video-wrap').style.display = 'block';
          document.getElementById('test-video-img').src = `/testador/video_feed?t=${Date.now()}`;
        } else {
          resultEl.textContent = data.erro || 'Falha ao conectar na camera.';
          resultEl.className = 'hint err';
        }
      } catch (_) {
        resultEl.textContent = 'Falha ao contatar o servidor.';
        resultEl.className = 'hint err';
      } finally {
        btn.disabled = false;
      }
    });

    document.getElementById('btn-video-parar').addEventListener('click', async () => {
      await fetch('/testador/video/parar', { method: 'POST' });
      document.getElementById('test-video-wrap').style.display = 'none';
      document.getElementById('test-video-img').src = '';
      const resultEl = document.getElementById('test-video-resultado');
      resultEl.textContent = 'Teste de video parado.';
      resultEl.className = 'hint';
    });

    carregarAmbiente();
    carregarProjetos();
    carregarModelosTestador();
  </script>
</body>
</html>
"""


def create_app(camera_manager: CameraManager, titulo: str) -> Flask:
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
            pp = create_project(
                str(data.get('nome', '')),
                data.get('classes_iniciais'),
                data.get('tipo', 'detect'),
            )
        except ValueError as exc:
            return jsonify({'ok': False, 'erro': str(exc)}), 400
        return jsonify({'ok': True, 'projeto': pp.nome})

    @app.get('/config/camera')
    def config_camera():
        cam = camera_manager.cameras.get(camera_manager.active_id) if camera_manager.active_id else None
        if not cam:
            return jsonify({'ip': '', 'usuario': '', 'senha': '', 'porta': '554', 'caminho': ''})
        return jsonify({
            'ip': cam['ip'],
            'usuario': cam['usuario'],
            'senha': cam['senha'],
            'porta': cam['porta'],
            'caminho': cam['caminho'],
        })

    @app.get('/cameras')
    def listar_cameras():
        return jsonify({'cameras': camera_manager.listar(), 'status': camera_manager.status()})

    @app.post('/cameras')
    def adicionar_camera():
        data = request.get_json(silent=True) or {}
        try:
            cam_id = camera_manager.adicionar(
                data.get('nome', ''), data.get('ip', ''), data.get('usuario', ''),
                data.get('senha', ''), data.get('porta', ''), data.get('caminho', ''),
            )
        except ValueError as exc:
            return jsonify({'ok': False, 'erro': str(exc)}), 400
        return jsonify({'ok': True, 'id': cam_id})

    @app.post('/cameras/<cam_id>/ativar')
    def ativar_camera(cam_id):
        try:
            camera_manager.ativar(cam_id)
        except (ValueError, RuntimeError) as exc:
            return jsonify({'ok': False, 'erro': str(exc)}), 400
        return jsonify({'ok': True})

    @app.post('/cameras/<cam_id>/remover')
    def remover_camera(cam_id):
        try:
            camera_manager.remover(cam_id)
        except ValueError as exc:
            return jsonify({'ok': False, 'erro': str(exc)}), 400
        return jsonify({'ok': True})

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
                frame = camera_manager.get_frame()
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
        frame = camera_manager.get_frame()
        if frame is None:
            return jsonify({'ok': False, 'erro': 'Sem frame disponivel da camera ainda.'}), 503
        nome = f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        destino = pp.capturas / nome
        if not cv2.imwrite(str(destino), frame):
            return jsonify({'ok': False, 'erro': 'Falha ao salvar a imagem.'}), 500
        return jsonify({'ok': True, 'arquivo': nome})

    @app.post('/imagens/upload')
    def upload_imagens():
        pp = get_project(request.form.get('projeto', ''))
        arquivos = request.files.getlist('arquivos')
        if not arquivos:
            return jsonify({'ok': False, 'erro': 'Nenhum arquivo enviado.'}), 400
        salvos = 0
        for i, arquivo in enumerate(arquivos):
            if not arquivo or not arquivo.filename:
                continue
            ext = Path(arquivo.filename).suffix.lower()
            if ext not in ('.jpg', '.jpeg', '.png'):
                continue
            nome = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{i}{ext}"
            arquivo.save(str(pp.capturas / nome))
            salvos += 1
        if not salvos:
            return jsonify({'ok': False, 'erro': 'Nenhuma imagem valida enviada (use .jpg/.jpeg/.png).'}), 400
        return jsonify({'ok': True, 'salvos': salvos})

    def _classes_no_label(lbl_path: Path) -> list:
        """Le so a 1a coluna (indice da classe) de cada linha do .txt de anotacao
        (formato YOLO, vale tanto pra caixa quanto pra poligono de segmentacao,
        que sempre comecam com o indice da classe) -- usado pra mostrar nas
        miniaturas quais classes aquela imagem tem, sem precisar abrir o modal
        de anotacao pra descobrir."""
        if not lbl_path.exists():
            return []
        indices = set()
        for linha in lbl_path.read_text(encoding='utf-8').splitlines():
            partes = linha.strip().split()
            if not partes:
                continue
            try:
                indices.add(int(partes[0]))
            except ValueError:
                continue
        return sorted(indices)

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
                info['classes'] = _classes_no_label(lbl)
            itens.append(info)
        return itens

    @app.get('/imagens')
    def listar_imagens():
        pp = get_project(request.args.get('projeto', ''))
        return jsonify({
            'pendentes': _listar(pp.capturas),
            'treino': _listar(pp.images_train, pp.labels_train),
            'validacao': _listar(pp.images_val, pp.labels_val),
            'task': load_project_task(pp),
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
        task = load_project_task(pp)
        caixas = []
        if lbl_dir is not None:
            lbl_path = lbl_dir / (Path(nome).stem + '.txt')
            if lbl_path.exists():
                for linha in lbl_path.read_text(encoding='utf-8').splitlines():
                    partes = linha.strip().split()
                    if not partes:
                        continue
                    try:
                        cls_id = int(partes[0])
                        valores = [float(v) for v in partes[1:]]
                    except ValueError:
                        continue
                    if task == 'segment':
                        if len(partes) < 7 or len(partes) % 2 == 0:
                            continue
                        pontos = [[valores[i], valores[i + 1]] for i in range(0, len(valores), 2)]
                        caixas.append({'classe': cls_id, 'pontos': pontos})
                    else:
                        if len(partes) != 5:
                            continue
                        xc, yc, w, h = valores
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
        task = load_project_task(pp)
        linhas = []
        for caixa in caixas:
            try:
                cls_id = int(caixa['classe'])
            except (KeyError, TypeError, ValueError):
                continue
            if not (0 <= cls_id < len(classes)):
                continue
            if task == 'segment':
                pontos = caixa.get('pontos')
                if not isinstance(pontos, list) or len(pontos) < 3:
                    continue
                try:
                    coords = [(min(max(float(p[0]), 0.0), 1.0), min(max(float(p[1]), 0.0), 1.0)) for p in pontos]
                except (TypeError, ValueError, IndexError):
                    continue
                valores = ' '.join(f'{x:.6f} {y:.6f}' for x, y in coords)
                linhas.append(f'{cls_id} {valores}')
            else:
                try:
                    x = float(caixa['x'])
                    y = float(caixa['y'])
                    w = float(caixa['w'])
                    h = float(caixa['h'])
                except (KeyError, TypeError, ValueError):
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

    @app.get('/treinamento/resultados/<run>/modelo/baixar')
    def baixar_modelo_resultado(run):
        """Baixa o best.pt de um treinamento concluido direto pelo navegador,
        sem precisar copiar o arquivo manualmente no servidor."""
        pp = get_project(request.args.get('projeto', ''))
        best = pp.runs / safe_name(run) / 'weights' / 'best.pt'
        if not best.exists():
            abort(404)
        nome_arquivo = f'{pp.nome}_{safe_name(run)}_best.pt'
        return send_file(best, as_attachment=True, download_name=nome_arquivo)

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

    @app.get('/testador/modelos')
    def testador_modelos():
        resultado = []
        for nome in list_projetos():
            pp = get_project(nome)
            if not pp.runs.exists():
                continue
            for d in sorted((p for p in pp.runs.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
                if (d / 'weights' / 'best.pt').exists():
                    resultado.append({
                        'projeto': nome,
                        'run': d.name,
                        'modificado': datetime.fromtimestamp(d.stat().st_mtime).strftime('%d/%m/%Y %H:%M'),
                        '_mtime': d.stat().st_mtime,
                    })
        resultado.sort(key=lambda r: r['_mtime'], reverse=True)
        for r in resultado:
            del r['_mtime']
        return jsonify(resultado)

    @app.post('/testador/imagem')
    def testador_imagem():
        pp = get_project(request.form.get('projeto', ''))
        run = safe_name(request.form.get('run', ''))
        best = pp.runs / run / 'weights' / 'best.pt'
        if not best.exists():
            return jsonify({'ok': False, 'erro': 'Modelo nao encontrado.'}), 404

        arquivo = request.files.get('imagem')
        if not arquivo or not arquivo.filename:
            return jsonify({'ok': False, 'erro': 'Nenhuma imagem enviada.'}), 400

        try:
            conf = float(request.form.get('conf', 0.4))
        except (TypeError, ValueError):
            conf = 0.4
        conf = min(max(conf, 0.05), 0.95)

        dados = arquivo.read()
        arr = np.frombuffer(dados, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'ok': False, 'erro': 'Nao foi possivel ler a imagem enviada.'}), 400

        try:
            modelo = carregar_modelo_cache(best)
            resultado = modelo.predict(frame, conf=conf, verbose=False)[0]
        except Exception as exc:
            return jsonify({'ok': False, 'erro': f'Falha ao rodar o modelo: {exc}'}), 500
        deteccoes = desenhar_deteccoes(frame, resultado)

        ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return jsonify({'ok': False, 'erro': 'Falha ao codificar imagem de resultado.'}), 500
        imagem_b64 = base64.b64encode(buf.tobytes()).decode('ascii')
        return jsonify({'ok': True, 'imagem': imagem_b64, 'deteccoes': deteccoes})

    @app.post('/testador/video/iniciar')
    def testador_video_iniciar():
        if test_video_state.running:
            return jsonify({'ok': False, 'erro': 'Ja ha um teste de video em andamento. Pare antes de iniciar outro.'}), 409

        data = request.get_json(silent=True) or {}
        try:
            rtsp_url = build_rtsp_url(
                data.get('ip', ''), data.get('usuario', ''), data.get('senha', ''),
                data.get('porta', '554'), data.get('caminho', ''),
            )
        except ValueError as exc:
            return jsonify({'ok': False, 'erro': str(exc)}), 400

        pp = get_project(str(data.get('projeto', '')))
        run = safe_name(str(data.get('run', '')))
        best = pp.runs / run / 'weights' / 'best.pt'
        if not best.exists():
            return jsonify({'ok': False, 'erro': 'Modelo nao encontrado.'}), 404

        try:
            conf = float(data.get('conf', 0.4))
        except (TypeError, ValueError):
            conf = 0.4
        conf = min(max(conf, 0.05), 0.95)

        stream_teste = CameraStream(rtsp_url)
        if not stream_teste.is_opened():
            stream_teste.cap.release()
            return jsonify({'ok': False, 'erro': 'Nao foi possivel conectar na camera. Confira os dados.'}), 400

        try:
            modelo = carregar_modelo_cache(best)
        except Exception as exc:
            stream_teste.cap.release()
            return jsonify({'ok': False, 'erro': f'Falha ao carregar o modelo: {exc}'}), 500

        stream_teste.start()
        test_video_state.start(stream_teste, pp.nome, run)
        threading.Thread(target=_loop_teste_video, args=(stream_teste, modelo, conf), daemon=True).start()
        return jsonify({'ok': True})

    @app.post('/testador/video/parar')
    def testador_video_parar():
        test_video_state.stop()
        return jsonify({'ok': True})

    @app.get('/testador/video/status')
    def testador_video_status():
        return jsonify(test_video_state.snapshot())

    @app.get('/testador/video_feed')
    def testador_video_feed():
        def generate():
            while test_video_state.running:
                jpeg = test_video_state.get_frame()
                if jpeg is None:
                    time.sleep(0.05)
                    continue
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
                time.sleep(0.03)
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    return app


def main() -> None:
    parser = load_config(CONFIG_FILE)

    camera_manager = CameraManager(parser)
    try:
        camera_manager.iniciar()
    except RuntimeError as exc:
        # Nao trava o programa por causa disso: o usuario pode cadastrar ou
        # ativar outra camera pela tela de Cameras assim que o app subir.
        print(f'[AVISO] {exc}')

    host = parser.get('web', 'host', fallback='0.0.0.0')
    port = parser.getint('web', 'porta', fallback=5011)
    titulo = parser.get('web', 'titulo', fallback='Treinamento de Modelo')

    app = create_app(camera_manager, titulo)
    try:
        print(f'Web de treinamento rodando em http://{host}:{port}')
        app.run(host=host, port=port, debug=False, threaded=True)
    finally:
        camera_manager.stop()


if __name__ == '__main__':
    main()
