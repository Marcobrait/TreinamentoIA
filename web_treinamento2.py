# python.exe -m venv .venv
# .\.venv\Scripts\Activate.ps1
# python -m pip install --upgrade pip
# python -m pip install -r requirements.txt

import base64
import configparser
import json
import shutil
import threading
import time
import zipfile
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
        # marcacoes previas das imagens pendentes (ex.: importadas do BeefVision,
        # feitas pela IA). So viram anotacao de verdade quando a pessoa revisa e
        # salva em treino/validacao.
        self.capturas_labels = self.dataset / 'capturas_labels'
        self.images_train = self.dataset / 'images' / 'train'
        self.images_val = self.dataset / 'images' / 'val'
        self.labels_train = self.dataset / 'labels' / 'train'
        self.labels_val = self.dataset / 'labels' / 'val'
        self.classes_file = self.dataset / 'classes.txt'
        self.task_file = self.dataset / 'task.txt'
        self.runs = self.root / 'runs_treinamento'

    def ensure(self) -> None:
        for d in (self.capturas, self.capturas_labels, self.images_train, self.images_val,
                  self.labels_train, self.labels_val, self.runs):
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


def rename_project(pp: ProjectPaths, novo_nome: str) -> ProjectPaths:
    slug = slugify_projeto(novo_nome)
    if not slug:
        raise ValueError('Nome do modelo nao pode ser vazio.')
    if slug == pp.nome:
        return pp
    destino = ProjectPaths(slug)
    if destino.root.exists():
        raise ValueError('Ja existe um modelo com esse nome.')
    try:
        pp.root.rename(destino.root)
    except OSError as exc:
        # No Windows, renomear uma pasta falha com "acesso negado" se
        # algum arquivo dela estiver com handle aberto em outro lugar (ex.:
        # um download do best.pt ainda em andamento, ou um antivirus
        # escaneando). Nao e algo que a gente controla aqui, mas precisa
        # virar uma mensagem clara em vez de um erro 500 cru.
        raise ValueError(
            'Nao foi possivel renomear -- algum arquivo desse modelo pode '
            'estar em uso em outro lugar (ex.: um download em andamento). Tente de novo em instantes.'
        ) from exc
    return destino


def duplicate_project(pp: ProjectPaths, novo_nome: str) -> ProjectPaths:
    slug = slugify_projeto(novo_nome)
    if not slug:
        raise ValueError('Nome do modelo nao pode ser vazio.')
    destino = ProjectPaths(slug)
    if destino.root.exists():
        raise ValueError('Ja existe um modelo com esse nome.')
    try:
        shutil.copytree(pp.root, destino.root)
    except OSError as exc:
        if destino.root.exists():
            shutil.rmtree(destino.root, ignore_errors=True)
        raise ValueError(f'Falha ao duplicar o modelo: {exc}') from exc
    return destino


def delete_project(pp: ProjectPaths) -> None:
    if not pp.root.exists():
        raise ValueError('Modelo nao encontrado.')
    try:
        shutil.rmtree(pp.root)
    except OSError as exc:
        raise ValueError(
            'Nao foi possivel excluir -- algum arquivo desse modelo pode '
            'estar em uso em outro lugar (ex.: um download em andamento). Tente de novo em instantes.'
        ) from exc


def read_run_args(run_dir: Path) -> dict:
    """Le args.yaml (escrito pelo ultralytics em todo treinamento) --
    parametros usados nesse treinamento especifico (epocas, batch, imgsz,
    modelo base etc.)."""
    args_path = run_dir / 'args.yaml'
    if not args_path.exists():
        return {}
    try:
        dados = yaml.safe_load(args_path.read_text(encoding='utf-8')) or {}
    except Exception:
        return {}
    chaves = ('epochs', 'batch', 'imgsz', 'model', 'task', 'optimizer', 'device')
    return {k: dados[k] for k in chaves if k in dados}


ARQ_TEMPOS = 'tempos_epocas.json'


def formatar_duracao(seg: float) -> str:
    """72 -> '1m 12s'; 3780 -> '1h 03m'; 9.4 -> '9.4s'."""
    seg = float(seg)
    if seg < 9.95:
        return f'{seg:.1f}s'
    if round(seg) < 60:  # arredonda antes de comparar: 59.6 vira "1m 00s", não "60s"
        return f'{round(seg)}s'
    m, s = divmod(int(round(seg)), 60)
    if m < 60:
        return f'{m}m {s:02d}s'
    h, m = divmod(m, 60)
    return f'{h}h {m:02d}m'


def resumo_tempos(epocas: list, epochs_total: int = 0, rodando: bool = False) -> dict:
    """Total, media, mais rapida/lenta e (durante o treino) estimativa do que falta."""
    if not epocas:
        return {'epocas': []}
    total = sum(epocas)
    media = total / len(epocas)
    resumo = {
        'epocas': [round(t, 1) for t in epocas],
        'total': round(total, 1),
        'media': round(media, 1),
        'mais_rapida': round(min(epocas), 1),
        'mais_lenta': round(max(epocas), 1),
    }
    if rodando and epochs_total > len(epocas):
        # media das ultimas 5 epocas: acompanha melhor mudancas de velocidade (cache, validacao...)
        recentes = epocas[-5:]
        resumo['restante'] = round(sum(recentes) / len(recentes) * (epochs_total - len(epocas)), 0)
    return resumo


def read_run_tempos(run_dir: Path) -> dict:
    """Tempo de cada epoca de um treinamento ja feito.
    1) tempos_epocas.json, gravado por este programa a cada epoca;
    2) se nao houver (treinos antigos), a coluna 'time' do results.csv do ultralytics,
       que e o tempo acumulado: a diferenca entre linhas e o tempo de cada epoca."""
    arq = run_dir / ARQ_TEMPOS
    if arq.exists():
        try:
            dados = json.loads(arq.read_text(encoding='utf-8'))
            epocas = [float(t) for t in dados.get('epocas', [])]
            if epocas:
                return resumo_tempos(epocas)
        except (ValueError, TypeError, OSError):
            pass
    csv_path = run_dir / 'results.csv'
    if not csv_path.exists():
        return {'epocas': []}
    try:
        linhas = csv_path.read_text(encoding='utf-8').strip().splitlines()
        cab = [c.strip() for c in linhas[0].split(',')]
        if 'time' not in cab:
            return {'epocas': []}
        i = cab.index('time')
        acumulado = [float(l.split(',')[i]) for l in linhas[1:] if l.strip()]
    except (ValueError, IndexError, OSError):
        return {'epocas': []}
    epocas = [b - a for a, b in zip([0.0] + acumulado[:-1], acumulado)]
    return resumo_tempos([t for t in epocas if t >= 0])


def read_run_metrics(run_dir: Path) -> dict:
    """Le a ULTIMA linha de results.csv (escrito pelo ultralytics a cada
    epoca) -- as metricas finais do treinamento (precisao, recall, mAP...).
    Sem pandas: so csv da biblioteca padrao mesmo, e 1 linha so que
    interessa."""
    csv_path = run_dir / 'results.csv'
    if not csv_path.exists():
        return {}
    try:
        linhas = csv_path.read_text(encoding='utf-8').strip().splitlines()
        if len(linhas) < 2:
            return {}
        cabecalho = [c.strip() for c in linhas[0].split(',')]
        valores = [v.strip() for v in linhas[-1].split(',')]
        bruto = dict(zip(cabecalho, valores))
    except Exception:
        return {}

    def pega(chave):
        v = bruto.get(chave)
        try:
            return round(float(v), 4) if v not in (None, '') else None
        except ValueError:
            return None

    metricas = {
        'epoca_final': pega('epoch'),
        'precisao': pega('metrics/precision(B)'),
        'recall': pega('metrics/recall(B)'),
        'map50': pega('metrics/mAP50(B)'),
        'map50_95': pega('metrics/mAP50-95(B)'),
    }
    # Projetos de segmentacao tem metricas extras de mascara (sufixo M).
    if 'metrics/precision(M)' in bruto:
        metricas['precisao_mascara'] = pega('metrics/precision(M)')
        metricas['recall_mascara'] = pega('metrics/recall(M)')
        metricas['map50_mascara'] = pega('metrics/mAP50(M)')
        metricas['map50_95_mascara'] = pega('metrics/mAP50-95(M)')
    return {k: v for k, v in metricas.items() if v is not None}


def project_summary(pp: ProjectPaths) -> dict:
    """Resumo de 1 projeto pra pagina 'Modelos' -- conta imagens, classes,
    treinamentos, sem precisar carregar nada pesado (nada de modelo YOLO
    aqui, so contagem de arquivos)."""
    def contar_imagens(pasta: Path) -> int:
        if not pasta.exists():
            return 0
        return sum(1 for ext in IMAGE_EXTENSIONS for _ in pasta.glob(ext))

    runs = []
    if pp.runs.exists():
        runs = [d for d in pp.runs.iterdir() if d.is_dir()]
    tem_modelo = any((d / 'weights' / 'best.pt').exists() for d in runs)

    caminhos_mtime = [pp.root] + runs
    ultima_modificacao = max((c.stat().st_mtime for c in caminhos_mtime if c.exists()), default=pp.root.stat().st_mtime)

    return {
        'nome': pp.nome,
        'task': load_project_task(pp),
        'total_classes': len(load_classes(pp)),
        'total_pendentes': contar_imagens(pp.capturas),
        'total_treino': contar_imagens(pp.images_train),
        'total_validacao': contar_imagens(pp.images_val),
        'total_treinamentos': len(runs),
        'tem_modelo': tem_modelo,
        'ultima_modificacao': datetime.fromtimestamp(ultima_modificacao).strftime('%d/%m/%Y %H:%M'),
        '_ultima_modificacao_ts': ultima_modificacao,
    }


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

    def editar(self, cam_id: str, nome: str, ip: str, usuario: str, senha: str, porta: str, caminho: str) -> None:
        """Atualiza os dados de uma camera ja cadastrada. Se for a camera
        ATIVA no momento, reconecta com os dados novos na hora -- senao a
        tela continuaria usando a conexao antiga ate reiniciar o
        programa."""
        if cam_id not in self.cameras:
            raise ValueError('Camera nao encontrada.')
        ip = (ip or '').strip()
        usuario = (usuario or '').strip()
        senha = (senha or '').strip()
        if not senha:
            # Campo de senha deixado em branco = mantem a senha ja salva
            # (assim nao precisa redigitar toda vez que so quer corrigir o
            # IP, por exemplo, nem a senha atual fica exposta na tela).
            senha = self.cameras[cam_id]['senha']
        if not ip or not usuario or not senha:
            raise ValueError('Preencha IP, usuario e senha da camera.')
        nome = (nome or '').strip() or f'Camera {cam_id}'
        porta = (porta or '554').strip() or '554'
        caminho = (caminho or '/cam/realmonitor?channel=1&subtype=0').strip()
        if not caminho.startswith('/'):
            caminho = '/' + caminho

        dados = {'nome': nome, 'ip': ip, 'usuario': usuario, 'senha': senha, 'porta': porta, 'caminho': caminho}
        self.cameras[cam_id] = dados
        try:
            update_config_section(CONFIG_FILE, f'camera_{cam_id}', dados)
        except Exception:
            pass

        if cam_id == self.active_id:
            rtsp = build_rtsp_url(ip, usuario, senha, porta, caminho)
            novo_stream = CameraStream(rtsp)
            if not novo_stream.is_opened():
                novo_stream.stop()
                raise RuntimeError(
                    'Os dados foram salvos, mas nao foi possivel reconectar com eles -- '
                    'confira IP, usuario, senha e porta.'
                )
            novo_stream.start()
            with self.lock:
                antiga = self.stream
                self.stream = novo_stream
            if antiga is not None:
                antiga.stop()

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


def pastas_de_marcacoes(pp: ProjectPaths) -> list:
    """Pastas com .txt de marcacao (formato YOLO): treino, validacao e pre-anotacoes das pendentes."""
    return [pp.labels_train, pp.labels_val, pp.capturas_labels]


def contar_uso_classes(pp: ProjectPaths) -> dict:
    """indice da classe -> {'marcacoes': quantas linhas, 'imagens': em quantas imagens aparece}."""
    uso = {}
    for pasta in pastas_de_marcacoes(pp):
        if not pasta.exists():
            continue
        for arq in pasta.glob('*.txt'):
            vistas = set()
            for linha in arq.read_text(encoding='utf-8', errors='replace').splitlines():
                partes = linha.split()
                if not partes:
                    continue
                try:
                    c = int(partes[0])
                except ValueError:
                    continue
                uso.setdefault(c, {'marcacoes': 0, 'imagens': 0})['marcacoes'] += 1
                vistas.add(c)
            for c in vistas:
                uso[c]['imagens'] += 1
    return uso


def remover_classe_do_dataset(pp: ProjectPaths, indice: int) -> dict:
    """Apaga as marcacoes da classe `indice` de todas as imagens e renumera as classes seguintes
    (a indice+1 vira indice, e assim por diante). Antes, guarda uma copia de seguranca das
    marcacoes e do classes.txt em dataset/_backup_classes/<data_hora>.zip."""
    agora = datetime.now().strftime('%Y%m%d_%H%M%S')
    pasta_backup = pp.dataset / '_backup_classes'
    pasta_backup.mkdir(parents=True, exist_ok=True)
    backup = pasta_backup / f'antes_de_remover_classe_{indice}_{agora}.zip'
    with zipfile.ZipFile(backup, 'w', zipfile.ZIP_DEFLATED) as z:
        if pp.classes_file.exists():
            z.write(pp.classes_file, 'classes.txt')
        for pasta in pastas_de_marcacoes(pp):
            if pasta.exists():
                for arq in pasta.glob('*.txt'):
                    z.write(arq, f'{pasta.name}/{arq.name}')

    removidas = afetadas = ficaram_vazias = 0
    for pasta in pastas_de_marcacoes(pp):
        if not pasta.exists():
            continue
        for arq in pasta.glob('*.txt'):
            originais = arq.read_text(encoding='utf-8', errors='replace').splitlines()
            novas, mudou, tirou = [], False, 0
            for linha in originais:
                partes = linha.split()
                try:
                    c = int(partes[0]) if partes else None
                except ValueError:
                    c = None
                if c is None:
                    novas.append(linha)
                elif c == indice:
                    tirou += 1
                    mudou = True
                elif c > indice:
                    novas.append(' '.join([str(c - 1)] + partes[1:]))
                    mudou = True
                else:
                    novas.append(linha)
            if not mudou:
                continue
            temporario = arq.with_suffix('.tmp')
            temporario.write_text('\n'.join(novas) + ('\n' if novas else ''), encoding='utf-8')
            temporario.replace(arq)  # troca de uma vez: nunca fica um arquivo pela metade
            if tirou:
                removidas += tirou
                afetadas += 1
                if not any(l.strip() for l in novas):
                    ficaram_vazias += 1

    classes = load_classes(pp)
    nome = classes.pop(indice)
    save_classes(pp, classes)
    return {'classe': nome, 'marcacoes_removidas': removidas, 'imagens_afetadas': afetadas,
            'imagens_sem_marcacao': ficaram_vazias, 'backup': str(backup)}


def safe_name(raw: str) -> str:
    return Path(str(raw)).name


def _normalizar_classe(nome: str) -> str:
    return ''.join(ch for ch in str(nome).lower() if ch.isalnum())


def converter_linha_yolo(linha: str, task: str, mapa_classes: dict):
    """Converte 1 linha de anotacao YOLO importada para o tipo deste projeto:
    - projeto de deteccao: poligono vira a caixa ao redor dele
    - projeto de segmentacao: caixa vira um poligono de 4 pontos
    mapa_classes: indice no arquivo importado -> indice neste projeto.
    Retorna a linha pronta ou None se for invalida."""
    partes = linha.strip().split()
    if len(partes) < 5:
        return None
    try:
        cls_origem = int(partes[0])
        v = [min(max(float(x), 0.0), 1.0) for x in partes[1:]]
    except ValueError:
        return None
    if cls_origem not in mapa_classes:
        return None
    cls_id = mapa_classes[cls_origem]
    if len(v) == 4:  # caixa: xc yc w h
        xc, yc, w, h = v
        if task == 'segment':
            x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            return f'{cls_id} ' + ' '.join(f'{x:.6f} {y:.6f}' for x, y in pts)
        return f'{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}'
    if len(v) >= 6 and len(v) % 2 == 0:  # poligono: x1 y1 x2 y2 ...
        if task == 'segment':
            return f'{cls_id} ' + ' '.join(f'{x:.6f}' for x in v)
        xs, ys = v[0::2], v[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        return f'{cls_id} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}'
    return None


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


def varinha_magica_contorno(img, seed_x: int, seed_y: int, raio_px: int, max_pontos: int = 80):
    """'Varinha magica': clica em cima do objeto e o contorno dele e
    detectado automaticamente. Por baixo, usa GrabCut (cv2.grabCut) em vez
    de um floodFill por tolerancia de cor.

    Por que GrabCut e nao floodFill: a 1a versao usava floodFill (feito pra
    parecer o pincel magico do Photoshop), mas em fotos reais com pouco
    contraste entre objeto e fundo, ou com varios objetos parecidos bem
    proximos (ex.: itens numa esteira/grade), qualquer tolerancia de cor
    acaba ou vazando pra imagem inteira, ou -- depois de limitarmos a area
    de busca como rede de seguranca -- preenchendo o recorte inteiro e
    devolvendo so um retangulo (a selecao literalmente virava "um
    quadrado", que foi o proximo bug reportado). GrabCut resolve isso
    porque nao decide pixel a pixel comparando com o vizinho: ele monta um
    MODELO ESTATISTICO de quais cores tendem a ser "objeto" e quais tendem
    a ser "fundo" dentro do recorte (a moldura fica marcada como fundo
    provavel, o miolo como objeto provavel) e refina isso em iteracoes --
    e assim que ferramentas de selecao inteligente de verdade (Selecionar
    Assunto do Photoshop, por ex.) costumam funcionar, nao com tolerancia
    fixa. Testado nos 3 cenarios que quebravam o floodFill (gradiente de
    luz, baixissimo contraste, objetos proximos numa grade) e acertou a
    area real do objeto com <2% de erro nos 3.

    raio_px: define o recorte (quadrado de 2*raio_px de lado, centrado no
    clique) onde o GrabCut roda. Ajuste pra ficar um pouco maior que o
    objeto que voce quer selecionar -- precisa sobrar uma margem ao redor
    do objeto (pro "com certeza fundo" abaixo ter onde marcar), entao um
    raio exageradamente justo ao objeto tende a falhar.

    A inicializacao usa uma MASCARA (nao so um retangulo) com 3 zonas:
    um circulo pequeno bem no ponto clicado marcado como "com certeza
    objeto" (a ancora mais forte que da pra passar pro GrabCut), uma faixa
    fina na borda do recorte marcada como "com certeza fundo", e o resto
    como "provavelmente objeto"/"provavelmente fundo". Isso ficou bem mais
    robusto a variacao no tamanho do raio do que inicializar so com um
    retangulo (testado varrendo o raio de 20 a 150px em 3 cenarios
    diferentes -- a versao so-retangulo funcionava so numa faixa
    estreitissima de raio quase exata, a versao com mascara funciona numa
    faixa ampla)."""
    h, w = img.shape[:2]
    x0 = max(0, seed_x - raio_px)
    y0 = max(0, seed_y - raio_px)
    x1 = min(w, seed_x + raio_px)
    y1 = min(h, seed_y + raio_px)
    recorte = img[y0:y1, x0:x1]
    rh, rw = recorte.shape[:2]
    if rh < 20 or rw < 20:
        return None

    margem = max(3, min(rh, rw) // 6)
    if rw - 2 * margem < 4 or rh - 2 * margem < 4:
        return None

    mask = np.full((rh, rw), cv2.GC_PR_BGD, np.uint8)
    mask[margem:rh - margem, margem:rw - margem] = cv2.GC_PR_FGD

    cx, cy = seed_x - x0, seed_y - y0
    raio_certeza = max(2, min(rh, rw) // 14)
    cv2.circle(mask, (cx, cy), raio_certeza, cv2.GC_FGD, -1)

    faixa = max(1, margem // 2)
    mask[:faixa, :] = cv2.GC_BGD
    mask[-faixa:, :] = cv2.GC_BGD
    mask[:, :faixa] = cv2.GC_BGD
    mask[:, -faixa:] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(recorte, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None

    mask_bin = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    if not mask_bin.any():
        return None

    kernel = np.ones((3, 3), np.uint8)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=1)

    contornos, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    maior = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(maior) < 30:
        return None

    perimetro = cv2.arcLength(maior, True)
    epsilon = max(1.0, perimetro * 0.004)
    simplificado = cv2.approxPolyDP(maior, epsilon, True)
    # Volta pra coordenadas da imagem original (desfaz o recorte).
    pontos = [(int(p[0][0]) + x0, int(p[0][1]) + y0) for p in simplificado]
    if len(pontos) < 3:
        return None

    # Poligono de segmentacao vira 1 linha so no .txt (classe + todos os
    # pontos) -- limita o tamanho pra nao gerar uma linha gigante.
    if len(pontos) > max_pontos:
        passo = len(pontos) / max_pontos
        pontos = [pontos[int(i * passo)] for i in range(max_pontos)]
    return pontos


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
        self.epoch_times = []      # segundos de cada epoca (treino + validacao)
        self._epoch_start = None
        self.historico = []        # [{'epoca': n, 'metricas': {...}}] para os graficos
        self._t_inicio = None      # time.time() do inicio / fim do treino (tempo decorrido)
        self._t_fim = None

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
            self.epoch_times = []
            self._epoch_start = None
            self.historico = []
            self._t_inicio = time.time()
            self._t_fim = None

    def mark_epoch_start(self) -> None:
        with self.lock:
            self._epoch_start = time.time()

    def mark_epoch_end(self):
        """Fecha a epoca atual e devolve quantos segundos ela levou (ou None)."""
        with self.lock:
            if self._epoch_start is None:
                return None
            dur = round(time.time() - self._epoch_start, 1)
            self.epoch_times.append(dur)
            self._epoch_start = None
            return dur

    def set_device(self, device: str) -> None:
        with self.lock:
            self.device = device

    def add_log(self, text: str) -> None:
        with self.lock:
            self.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")
            self.log = self.log[-500:]

    def update_epoch(self, epoch: int, epochs_total: int, metrics: dict) -> None:
        with self.lock:
            self.epoch = epoch
            self.epochs_total = epochs_total
            self.metrics = metrics
            # o ultralytics chama o fim de epoca mais uma vez no final (validacao com o best.pt):
            # o grafico fica com a primeira leitura de cada epoca
            if not self.historico or self.historico[-1]['epoca'] < epoch:
                self.historico.append({'epoca': epoch, 'metricas': dict(metrics)})

    def finish(self, run_dir) -> None:
        with self.lock:
            self.running = False
            self.status = 'concluido'
            self._t_fim = time.time()
            self.run_dir = str(run_dir) if run_dir else None
            self.finished_at = datetime.now().strftime('%d/%m/%Y %H:%M:%S')

    def fail(self, error) -> None:
        with self.lock:
            self.running = False
            self.status = 'erro'
            self._t_fim = time.time()
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
                'log': list(self.log),
                'run_dir': self.run_dir,
                'error': self.error,
                'started_at': self.started_at,
                'finished_at': self.finished_at,
                'tempos': resumo_tempos(self.epoch_times, self.epochs_total, self.running),
                'historico': list(self.historico),
                # tempo desde o inicio do treino e da epoca que esta rodando agora
                'decorrido': round((self._t_fim or time.time()) - self._t_inicio, 1) if self._t_inicio else None,
                'epoca_atual_seg': round(time.time() - self._epoch_start, 1)
                if self.running and self._epoch_start else None,
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

        def on_train_epoch_start(trainer):
            training_state.mark_epoch_start()

        def on_fit_epoch_end(trainer):
            # fim da epoca (treino + validacao): mede o tempo e grava junto do treino
            duracao = training_state.mark_epoch_end()
            epoch = int(getattr(trainer, 'epoch', 0)) + 1
            epochs_total = int(getattr(trainer, 'epochs', params['epocas']))
            metrics = _extract_metrics(trainer)
            training_state.update_epoch(epoch, epochs_total, metrics)
            try:
                destino = Path(getattr(trainer, 'save_dir', run_dir)) / ARQ_TEMPOS
                destino.write_text(json.dumps({'epocas': training_state.snapshot()['tempos']['epocas']}), encoding='utf-8')
            except OSError:
                pass
            resumo = ', '.join(f'{k}={v}' for k, v in metrics.items())
            tempo = f' ({formatar_duracao(duracao)})' if duracao is not None else ''
            training_state.add_log(f'Epoca {epoch}/{epochs_total}{tempo} - {resumo}')

        model.add_callback('on_train_start', on_train_start)
        model.add_callback('on_train_epoch_start', on_train_epoch_start)
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
  <title>Treinamento de Modelo - Contador IA (Varinha Magica)</title>
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
      position: relative; flex: 1 1 640px; min-width: 320px; max-width: 100%; border-radius: var(--radius-md);
      overflow: hidden; background: #000; box-shadow: var(--shadow-sm); line-height: 0;
    }
    .video-frame img { width: 100%; max-width: 100%; display: block; transition: width .12s linear; }
    .video-frame img.video-pausado { filter: grayscale(1); }
    .video-pause-overlay {
      position: absolute; inset: 0; display: none; align-items: center; justify-content: center;
      background: rgba(0,0,0,.2); pointer-events: none;
    }
    .video-pause-overlay svg { width: 56px; height: 56px; color: #fff; filter: drop-shadow(0 2px 8px rgba(0,0,0,.6)); }
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
    /* classes do modelo: botao da classe + editar / remover */
    .chip-classe { display: inline-flex; align-items: center; gap: 2px; }
    .chip-classe .chip-select small { font-weight: 400; color: var(--text-muted); margin-left: 6px; font-variant-numeric: tabular-nums; }
    .chip-acao {
      width: 26px; height: 26px; border: 0; border-radius: 50%; background: none; color: var(--text-muted);
      cursor: pointer; display: inline-grid; place-items: center; font-size: 13px; line-height: 1; padding: 0;
    }
    .chip-acao:hover { background: var(--fill-strong); color: var(--text); }
    .chip-acao.remover:hover { background: rgba(220,38,38,.1); color: #dc2626; }

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
    .thumb-badge.pre { color: #8ec5ff; }
    .boxes-list li.sel { background: var(--accent-tint); border-radius: 6px; }
    .boxes-list li { padding: 3px 6px; cursor: pointer; }
    .boxes-list select { font: inherit; font-size: 12px; padding: 2px 4px; border-radius: 6px; border: 1px solid var(--border); background: var(--fill); color: var(--text); max-width: 170px; }
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

    .thumb.selecionavel { cursor: pointer; }
    .thumb.selecionavel img { user-select: none; -webkit-user-select: none; }
    .thumb.selecionado { box-shadow: 0 0 0 3px var(--accent); }
    .thumb.selecionado img { opacity: .82; }
    .thumb-select-dot {
      position: absolute; top: 8px; left: 8px; width: 22px; height: 22px; border-radius: 50%;
      background: rgba(0,0,0,.35); border: 2px solid rgba(255,255,255,.9); z-index: 2;
      display: flex; align-items: center; justify-content: center;
      -webkit-backdrop-filter: blur(4px); backdrop-filter: blur(4px);
      transition: background-color .15s var(--ease), border-color .15s var(--ease);
    }
    .thumb-select-dot svg { width: 13px; height: 13px; color: #fff; opacity: 0; transition: opacity .12s var(--ease); }
    .thumb.selecionado .thumb-select-dot { background: var(--accent); border-color: var(--accent); }
    .thumb.selecionado .thumb-select-dot svg { opacity: 1; }

    .selecao-barra {
      position: fixed; left: 0; right: 0; bottom: 0; z-index: 60;
      display: none; align-items: center; justify-content: space-between; gap: 16px;
      padding: 14px 24px; background: var(--card); border-top: 1px solid var(--border);
      box-shadow: 0 -8px 24px rgba(0,0,0,.14);
    }
    .selecao-barra.ativa { display: flex; }
    .selecao-barra-contagem { font-size: 13px; font-weight: 600; color: var(--text-muted); }
    .selecao-barra-acoes { display: flex; gap: 10px; }
    .selecao-barra-acoes .btn-primary:disabled { opacity: .45; cursor: not-allowed; }

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
    /* tempo de cada epoca */
    .tempos-epocas { margin-top: 12px; }
    .tempos-resumo { display: flex; flex-wrap: wrap; gap: 8px; }
    .tempos-resumo .metric-pill b { color: var(--text); font-weight: 600; }
    .tempos-lista {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(92px, 1fr)); gap: 4px 10px;
      margin-top: 8px; max-height: 132px; overflow-y: auto; font-size: 11.5px; color: var(--text-muted);
      font-variant-numeric: tabular-nums;
    }
    .tempos-lista span { display: flex; justify-content: space-between; gap: 6px; padding: 2px 0; border-bottom: 1px solid var(--border); }
    .tempos-lista span b { color: var(--text); font-weight: 500; }
    .tempos-lista span.lenta b { color: #d97706; }
    .tempos-titulo { font-size: 11.5px; font-weight: 600; color: var(--text-muted); margin-top: 10px; }
    .log-box {
      margin-top: 12px; background: #1d1d1f; border-radius: var(--radius-md);
      padding: 14px; font-size: 11.5px; color: #c8c8cd; max-height: 180px; overflow-y: auto;
      white-space: pre-wrap; font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    }

    /* ================= Treinamento (layout em cartoes) ================= */
    .tr-card { padding: 22px 24px; }
    .tr-cab { display: flex; align-items: center; flex-wrap: wrap; gap: 6px 10px; margin-bottom: 16px; }
    .tr-cab h2 { margin: 0; font-size: 17px; white-space: nowrap; }
    .tr-cab h2 small { font-size: 14px; font-weight: 600; color: var(--text-muted); margin-left: 4px; }
    .tr-cab .tr-dir { margin-left: auto; font-size: 12.5px; color: var(--text-muted); }
    .tr-ico { display: inline-flex; width: 22px; height: 22px; color: var(--c, var(--accent)); flex: 0 0 22px; }
    .tr-ico svg, .tr-btn-iniciar svg, .tr-btn-mini svg, .tr-stat svg, .tr-tile-ico svg { width: 100%; height: 100%; }

    .tr-form { display: grid; grid-template-columns: repeat(4, minmax(110px, 1fr)) minmax(220px, 2fr) auto; gap: 14px; align-items: end; }
    .tr-form label { display: flex; flex-direction: column; gap: 6px; font-size: 13px; font-weight: 600; color: var(--text); }
    .tr-form input, .tr-form select {
      height: 44px; background: #fff; border: 1px solid var(--border); border-radius: var(--radius-sm);
      padding: 0 12px; color: var(--text); font-size: 14px; width: 100%;
      transition: border-color .15s var(--ease), box-shadow .15s var(--ease);
    }
    .tr-form input:focus, .tr-form select:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 4px var(--accent-tint); }
    .tr-btn-iniciar {
      height: 52px; padding: 0 26px; border: 0; border-radius: var(--radius-md); background: #16a34a; color: #fff;
      font-size: 15px; font-weight: 600; display: inline-flex; align-items: center; gap: 10px; cursor: pointer;
      box-shadow: 0 4px 14px rgba(22,163,74,.3); transition: background-color .15s var(--ease), transform .1s var(--ease);
    }
    .tr-btn-iniciar svg { width: 18px; height: 18px; }
    .tr-btn-iniciar:hover { background: #15803d; }
    .tr-btn-iniciar:active { transform: scale(.98); }
    .tr-btn-iniciar:disabled { background: #9ca3af; box-shadow: none; cursor: not-allowed; }
    .tr-dica { display: flex; gap: 8px; align-items: flex-start; margin-top: 14px; font-size: 12.5px; color: var(--text-muted); }
    .tr-dica .tr-ico { width: 16px; height: 16px; flex-basis: 16px; margin-top: 1px; }
    #ambiente-info { margin-top: 6px; padding-left: 24px; }

    .tr-painel { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.25fr); gap: 24px; }
    .tr-painel > .tr-log { grid-column: 1 / -1; }
    @media (max-width: 1100px) {
      .tr-painel { grid-template-columns: minmax(0, 1fr); }
      .tr-form { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .tr-form label.tr-larga, .tr-form .tr-btn-iniciar { grid-column: 1 / -1; }
    }

    .tr-epoca { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 10px; }
    .tr-epoca b { font-size: 19px; font-weight: 600; }
    .tr-epoca span { font-size: 30px; font-weight: 700; color: var(--accent); font-variant-numeric: tabular-nums; }
    .tr-barra { height: 12px; background: var(--fill-strong); border-radius: 999px; overflow: hidden; }
    .tr-barra i { display: block; height: 100%; width: 0; background: var(--accent); border-radius: 999px; transition: width .5s var(--ease); }
    .tr-barra.ok i { background: #16a34a; }
    .tr-barra.erro i { background: var(--err, #dc2626); }
    .tr-msg { margin-top: 12px; font-size: 13px; border-radius: var(--radius-sm); padding: 8px 12px; }
    .tr-msg.ok { background: rgba(22,163,74,.1); color: #166534; }
    .tr-msg.erro { background: rgba(220,38,38,.08); color: #991b1b; }
    .tr-msg:empty { display: none; }
    .tr-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 0 24px; margin-top: 18px; }
    .tr-stat { display: grid; grid-template-columns: 22px 1fr auto; gap: 10px; align-items: center; padding: 11px 0; border-top: 1px solid var(--fill-strong); }
    .tr-stat svg { width: 20px; height: 20px; color: var(--text-muted); }
    .tr-stat span { font-size: 13px; color: var(--text-muted); }
    .tr-stat b { font-size: 14px; font-weight: 600; font-variant-numeric: tabular-nums; }

    .tr-tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(118px, 1fr)); gap: 10px; }
    .tr-tile {
      display: flex; gap: 10px; align-items: center; padding: 12px; border-radius: var(--radius-md);
      background: color-mix(in srgb, var(--c) 7%, #fff); border: 1px solid color-mix(in srgb, var(--c) 22%, #fff);
    }
    .tr-tile-ico { width: 30px; height: 30px; flex: 0 0 30px; border-radius: 50%; display: grid; place-items: center; color: var(--c); background: color-mix(in srgb, var(--c) 14%, #fff); }
    .tr-tile-ico svg { width: 17px; height: 17px; }
    .tr-tile span { display: block; font-size: 11.5px; color: var(--text-muted); }
    .tr-tile b { display: block; font-size: 17px; font-weight: 700; font-variant-numeric: tabular-nums; color: var(--text); }
    .tr-vazio { color: var(--text-muted); font-size: 13px; padding: 18px 0; text-align: center; }

    .tr-seg { margin-left: auto; display: inline-flex; background: var(--fill); border-radius: 999px; padding: 3px; }
    .tr-seg button { border: 0; background: none; border-radius: 999px; padding: 5px 14px; font-size: 12.5px; font-weight: 500; color: var(--text-muted); cursor: pointer; }
    .tr-seg button.ativo { background: #fff; color: var(--text); box-shadow: var(--shadow-sm); }
    .tr-legenda { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12.5px; color: var(--text-muted); margin-bottom: 6px; }
    .tr-legenda span { display: inline-flex; align-items: center; gap: 6px; }
    .tr-legenda i { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
    .tr-graf-area { position: relative; }
    .tr-graf-area svg { display: block; width: 100%; }
    .tr-graf-area .eixo { font-size: 11px; fill: var(--text-muted); }
    .tr-dica-graf {
      position: absolute; pointer-events: none; background: #1d1d1f; color: #fff; border-radius: 8px; padding: 8px 10px;
      font-size: 12px; line-height: 1.55; white-space: nowrap; z-index: 3; box-shadow: var(--shadow);
    }
    .tr-dica-graf i { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
    .tr-dica-graf[hidden] { display: none; }

    .tr-btn-mini {
      margin-left: auto; border: 1px solid var(--border); background: #fff; border-radius: var(--radius-sm);
      padding: 6px 12px; font-size: 12.5px; font-weight: 500; color: var(--text); cursor: pointer; display: inline-flex; align-items: center; gap: 6px;
    }
    .tr-btn-mini:hover { background: var(--fill); }
    .tr-btn-mini svg { width: 14px; height: 14px; }
    .tr-ep-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(84px, 1fr)); gap: 8px; }
    .tr-ep-grid.todas { max-height: 330px; overflow-y: auto; padding-right: 4px; }
    .tr-ep { border: 1px solid var(--fill-strong); border-radius: var(--radius-sm); padding: 7px 9px; background: #fff; }
    .tr-ep span { display: block; font-size: 11px; color: var(--text-muted); }
    .tr-ep b { font-size: 13.5px; font-weight: 600; font-variant-numeric: tabular-nums; }
    .tr-ep.lenta { border-color: rgba(217,119,6,.45); background: rgba(217,119,6,.06); }
    .tr-ep.lenta b { color: #b45309; }

    .tr-log { background: #15171c; border-radius: var(--radius); box-shadow: var(--shadow); overflow: hidden; }
    .tr-log-cab { display: flex; align-items: center; gap: 10px; padding: 14px 18px; border-bottom: 1px solid rgba(255,255,255,.08); color: #f3f4f6; }
    .tr-log-cab h2 { margin: 0; font-size: 15px; font-weight: 600; color: #f3f4f6; }
    .tr-log-cab .tr-ico { color: #f3f4f6; width: 18px; height: 18px; flex-basis: 18px; }
    .tr-log-acoes { margin-left: auto; display: flex; gap: 8px; align-items: center; }
    .tr-log-acoes button {
      border: 1px solid rgba(255,255,255,.15); background: rgba(255,255,255,.04); color: #e5e7eb; border-radius: var(--radius-sm);
      padding: 6px 12px; font-size: 12.5px; cursor: pointer; display: inline-flex; align-items: center; gap: 6px;
    }
    .tr-log-acoes button:hover { background: rgba(255,255,255,.1); }
    .tr-log-acoes button svg { width: 14px; height: 14px; }
    .tr-switch { display: inline-flex; align-items: center; gap: 8px; font-size: 12.5px; color: #e5e7eb; cursor: pointer; user-select: none; }
    .tr-switch input { display: none; }
    .tr-switch i { width: 34px; height: 20px; border-radius: 999px; background: #4b5563; position: relative; transition: background-color .15s; }
    .tr-switch i::after { content: ""; position: absolute; top: 3px; left: 3px; width: 14px; height: 14px; border-radius: 50%; background: #fff; transition: transform .15s; }
    .tr-switch input:checked + i { background: var(--accent); }
    .tr-switch input:checked + i::after { transform: translateX(14px); }
    .tr-log-corpo {
      max-height: 260px; overflow-y: auto; padding: 12px 18px; font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      font-size: 12px; line-height: 1.7; color: #d1d5db;
    }
    .tr-log-corpo div { white-space: pre-wrap; word-break: break-word; }
    .tr-log-corpo .h { color: #6b7280; }
    .tr-log-corpo .n { font-weight: 700; margin: 0 8px; }
    .tr-log-corpo .n.info { color: #22d3ee; }
    .tr-log-corpo .n.aviso { color: #fbbf24; }
    .tr-log-corpo .n.erro { color: #f87171; }
    .tr-log-corpo .ep { color: #a5f3fc; }
    .tr-log-vazio { color: #6b7280; }

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
    .canvas-nav-row {
      display: flex; align-items: center; justify-content: center; gap: 10px;
      height: min(70vh, 640px);
    }
    .nav-arrow { flex: 0 0 auto; width: 34px; height: 34px; font-size: 18px; border-radius: 999px; }
    .nav-arrow:disabled { opacity: .3; cursor: not-allowed; }
    #anotacao-canvas { flex: 0 1 auto; min-width: 0; max-width: 100%; max-height: 100%; border-radius: var(--radius-md); background: #000; cursor: crosshair; }
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

    /* ---- pagina Modelos ---- */
    .modelos-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(270px, 1fr)); gap: 16px; }
    .modelo-card {
      background: var(--fill); border-radius: var(--radius-md); padding: 18px;
      cursor: pointer; transition: transform .18s var(--ease), box-shadow .18s var(--ease);
      display: flex; flex-direction: column; gap: 12px;
    }
    .modelo-card:hover { transform: translateY(-2px); box-shadow: var(--shadow); }
    .modelo-card-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; }
    .modelo-card-header strong { font-size: 15px; font-weight: 650; letter-spacing: -.01em; line-height: 1.3; }
    .modelo-badge {
      font-size: 10.5px; font-weight: 700; letter-spacing: .02em; padding: 3px 9px; border-radius: 999px;
      white-space: nowrap; flex: 0 0 auto;
    }
    .modelo-badge.detect { background: rgba(0,113,227,.1); color: var(--accent); }
    .modelo-badge.segment { background: rgba(168,85,247,.12); color: #a855f7; }
    .modelo-stats-mini { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
    .modelo-stat-mini { text-align: center; background: rgba(0,0,0,.03); border-radius: var(--radius-sm); padding: 8px 4px; }
    .modelo-stat-mini strong { display: block; font-size: 16px; font-weight: 650; }
    .modelo-stat-mini span { font-size: 10px; color: var(--text-muted); }
    .modelo-card-footer {
      display: flex; align-items: center; justify-content: space-between; gap: 8px;
      margin-top: auto; padding-top: 10px; border-top: 1px solid var(--border);
    }
    .modelo-card-footer .hint { font-size: 11px; }
    .modelo-card-actions { display: flex; gap: 4px; }
    .modelo-card-actions button {
      background: none; border: none; color: var(--text-muted); cursor: pointer;
      width: 26px; height: 26px; border-radius: var(--radius-sm); font-size: 13px;
      display: flex; align-items: center; justify-content: center;
      transition: background-color .15s var(--ease), color .15s var(--ease);
    }
    .modelo-card-actions button:hover { background: var(--fill-strong); color: var(--text); }
    .modelo-card-actions button.danger:hover { background: rgba(239,68,68,.1); color: var(--err); }

    .detalhe-topo { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; margin-bottom: 18px; }
    .detalhe-titulo-row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
    .detalhe-titulo-row h2 { margin: 0; font-size: 22px; font-weight: 650; letter-spacing: -.015em; }
    .detalhe-layout { display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap; }
    .detalhe-main { flex: 1 1 560px; min-width: 0; display: flex; flex-direction: column; gap: 18px; }
    .stat-panel { flex: 0 0 300px; min-width: 260px; display: flex; flex-direction: column; gap: 14px; }
    .stat-group { background: var(--fill); border-radius: var(--radius-md); overflow: hidden; }
    .stat-group-header {
      background: rgba(0,0,0,.055); padding: 9px 14px; font-size: 10.5px; font-weight: 700;
      letter-spacing: .06em; text-transform: uppercase; color: var(--text-muted);
    }
    .stat-grid-2 { display: grid; grid-template-columns: 1fr 1fr; padding: 4px 6px 10px; }
    .stat-cell { display: flex; justify-content: space-between; align-items: baseline; padding: 6px 8px; font-size: 12.5px; gap: 6px; }
    .stat-cell .stat-label { color: var(--text-muted); white-space: nowrap; }
    .stat-cell .stat-value { font-weight: 600; text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .stat-rows { padding: 4px 14px 10px; }
    .stat-row { display: flex; align-items: center; gap: 8px; padding: 7px 0; font-size: 12.5px; }
    .stat-row .stat-dot { width: 8px; height: 8px; border-radius: 50%; flex: 0 0 auto; }
    .stat-row .stat-label { flex: 1 1 auto; color: var(--text); }
    .stat-row .stat-value { font-weight: 600; }

    .run-card { background: var(--fill); border-radius: var(--radius-md); padding: 16px; }
    .run-card-header { display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
    .run-card-header strong { font-size: 14px; font-weight: 600; }
    .run-metric-pills { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
    .run-metric-pill { font-size: 11px; font-weight: 600; background: rgba(0,113,227,.08); color: var(--accent); border-radius: 999px; padding: 3px 10px; }
    .run-gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 8px; }
    .run-gallery img {
      width: 100%; height: 90px; object-fit: cover; border-radius: var(--radius-sm); cursor: pointer;
      background: #000; display: block; transition: transform .15s var(--ease), box-shadow .15s var(--ease);
    }
    .run-gallery img:hover { transform: translateY(-2px); box-shadow: var(--shadow-sm); }

    .lightbox-box { max-width: 1240px; width: 96vw; }
    .lightbox-body { display: flex; gap: 0; padding: 0; flex-wrap: nowrap; }
    .lightbox-image-wrap {
      flex: 1 1 auto; position: relative; background: #101012; display: flex;
      align-items: center; justify-content: center; min-height: 420px; max-height: 80vh; overflow: hidden;
    }
    .lightbox-image-wrap img { max-width: 100%; max-height: 80vh; object-fit: contain; display: block; }
    .lightbox-nav {
      position: absolute; top: 50%; transform: translateY(-50%); background: rgba(0,0,0,.45); color: #fff;
      border: none; width: 38px; height: 38px; border-radius: 50%; font-size: 20px; cursor: pointer;
      display: flex; align-items: center; justify-content: center; transition: background-color .15s var(--ease);
    }
    .lightbox-nav:hover { background: rgba(0,0,0,.7); }
    .lightbox-nav:disabled { opacity: .25; cursor: not-allowed; }
    .lightbox-nav.prev { left: 14px; }
    .lightbox-nav.next { right: 14px; }
    .lightbox-counter {
      position: absolute; bottom: 12px; right: 16px; color: rgba(255,255,255,.85); font-size: 11.5px;
      background: rgba(0,0,0,.45); padding: 3px 10px; border-radius: 999px;
    }
    .lightbox-side { flex: 0 0 300px; padding: 18px; overflow-y: auto; max-height: 80vh; display: flex; flex-direction: column; gap: 14px; }
    @media (max-width: 860px) { .lightbox-body { flex-direction: column; } .lightbox-side { flex: 0 0 auto; max-height: none; } }
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
      <button type="button" class="side-btn" data-view="modelos">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3.5" y="3.5" width="7" height="7" rx="1.3"/><rect x="13.5" y="3.5" width="7" height="7" rx="1.3"/><rect x="3.5" y="13.5" width="7" height="7" rx="1.3"/><rect x="13.5" y="13.5" width="7" height="7" rx="1.3"/></svg>
        <span class="label-text">Modelos</span>
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
          <div id="video-pause-overlay" class="video-pause-overlay">
            <svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1"></rect><rect x="14" y="5" width="4" height="14" rx="1"></rect></svg>
          </div>
        </div>
        <div class="capture-panel">
          <div class="panel-block">
            <span class="panel-label" title="Pausa so o video ao vivo (sem gastar rede enviando frames pro navegador) -- capturar frame e treinar continuam funcionando normalmente, mesmo com o video parado.">Video ao vivo</span>
            <button type="button" id="btn-video-toggle" class="btn-secondary btn-block">Parar video</button>
            <div id="video-status" class="hint"></div>
          </div>

          <div class="panel-divider"></div>

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

          <div class="panel-divider"></div>

          <div class="panel-block">
            <span class="panel-label" title="Arquivo .zip gerado no BeefVision (Exportar p/ Treinamento): imagens + marcacoes da IA no padrao YOLO">Importar imagens ja marcadas (.zip do BeefVision)</span>
            <label class="file-picker" for="input-importar-zip">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v11M7.5 10.5 12 15l4.5-4.5"/><path d="M4 15.5V18a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2.5"/></svg>
              <span id="importar-picker-texto">Escolher arquivo .zip</span>
            </label>
            <input type="file" id="input-importar-zip" accept=".zip,application/zip" style="display:none" />
            <div class="radio-row" title="Pendentes: voce revisa cada imagem (a marcacao da IA ja aparece) e salva em treino/validacao">
              <label><input type="radio" name="importar-destino" value="pendentes" checked /> Pendentes (revisar)</label>
              <label><input type="radio" name="importar-destino" value="treino" /> Treino</label>
              <label><input type="radio" name="importar-destino" value="validacao" /> Validacao</label>
            </div>
            <button id="btn-importar-zip" class="btn-secondary btn-block">Importar</button>
            <div id="importar-resultado" class="hint"></div>
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
        <button type="button" id="btn-modo-selecao" class="btn-secondary">Selecionar</button>
      </div>
      <div id="filtro-classes-row" class="panel-block" style="display:none; margin-bottom:14px;">
        <div style="display:flex; align-items:center;">
          <span class="panel-label">Filtrar por classe</span>
          <button type="button" id="btn-limpar-filtro-classes" class="link-btn-sm" style="display:none;">Limpar filtro</button>
        </div>
        <div id="filtro-classes-chips" class="chip-list"></div>
      </div>
      <div id="galeria-contagem" class="hint" style="display:none; margin-bottom:8px;"></div>
      <div id="galeria" class="gallery"></div>
    </section>

    <section class="card tr-card" id="tr-config">
      <div class="tr-cab"><span class="tr-ico"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h0a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v0a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg></span><h2>Configuração do Treinamento</h2></div>
      <form id="form-treinamento" class="tr-form">
        <label>Épocas
          <input type="number" id="cfg-epocas" value="100" min="1" max="2000" />
        </label>
        <label>Tamanho da imagem
          <input type="number" id="cfg-imgsz" value="640" min="32" max="1920" step="32" />
        </label>
        <label>Batch
          <select id="cfg-batch">
            <option value="auto">Automático</option>
            <option value="4">4</option>
            <option value="8" selected>8</option>
            <option value="16">16</option>
            <option value="32">32</option>
            <option value="48">48</option>
            <option value="64">64</option>
          </select>
        </label>
        <label>Workers
          <input type="number" id="cfg-workers" value="4" min="0" />
        </label>
        <label class="tr-larga">Modelo base
          <select id="cfg-modelo">
            <option value="yolov8n.pt" selected>YOLOv8n (rapido, leve)</option>
            <option value="yolov8s.pt">YOLOv8s (equilibrado, precisa internet)</option>
            <option value="yolov8m.pt">YOLOv8m (mais preciso, precisa internet)</option>
            <option value="__ultimo__">Continuar do ultimo treinamento deste modelo</option>
          </select>
        </label>
        <button type="submit" id="btn-iniciar-treino" class="tr-btn-iniciar"><svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 4.5v15l13-7.5z"/></svg>Iniciar Treinamento</button>
      </form>
      <div class="tr-dica"><span class="tr-ico"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/></svg></span>
        <span>Workers controla quantos processos carregam/preparam imagens em paralelo. Valores mais altos usam mais RAM
        e podem ser mais rápidos; se o treino travar, reduza para 2, 1 ou 0.</span></div>
      <div id="ambiente-info" class="hint">Verificando GPU...</div>
    </section>

    <div id="treino-progresso" class="tr-painel" style="display:none;">
      <section class="card tr-card">
        <div class="tr-cab"><span class="tr-ico"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/></svg></span><h2>Progresso do Treinamento</h2><span class="tr-dir" id="tr-device"></span></div>
        <div class="tr-epoca"><b id="tr-epoca-txt">Aguardando...</b><span id="tr-pct">0%</span></div>
        <div class="tr-barra" id="tr-barra"><i id="tr-barra-fill"></i></div>
        <div class="tr-msg" id="tr-msg"></div>
        <div class="tr-stats">
          <div class="tr-stat"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg><span>Tempo total</span><b id="tr-st-total">-</b></div>
          <div class="tr-stat"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="13" r="8"/><path d="M12 9v4l2.5 2M9 2h6"/></svg><span>Época atual</span><b id="tr-st-atual">-</b></div>
          <div class="tr-stat"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 3h12M6 21h12M7 3c0 5 10 5 10 9s-10 4-10 9M17 3c0 5-10 5-10 9s10 4 10 9"/></svg><span>Tempo por época (média)</span><b id="tr-st-media">-</b></div>
          <div class="tr-stat"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 3h12M6 21h12M7 3c0 5 10 5 10 9s-10 4-10 9M17 3c0 5-10 5-10 9s10 4 10 9"/></svg><span>Tempo restante (aprox.)</span><b id="tr-st-restante">-</b></div>
        </div>
      </section>

      <section class="card tr-card">
        <div class="tr-cab"><span class="tr-ico"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/></svg></span><h2>Métricas Atuais <small id="tr-metr-epoca"></small></h2></div>
        <div class="tr-tiles" id="tr-tiles"><div class="tr-vazio">Aguardando a primeira época...</div></div>
      </section>

      <section class="card tr-card">
        <div class="tr-cab"><span class="tr-ico"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17l6-6 4 4 8-8M15 7h6v6"/></svg></span><h2>Gráficos de Treinamento</h2>
          <div class="tr-seg" id="tr-graf-modo">
            <button type="button" data-modo="metricas" class="ativo">Métricas</button>
            <button type="button" data-modo="perdas">Perdas</button>
          </div>
        </div>
        <div class="tr-legenda" id="tr-legenda"></div>
        <div class="tr-graf-area" id="tr-graf"></div>
      </section>

      <section class="card tr-card">
        <div class="tr-cab"><span class="tr-ico"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg></span><h2>Tempo de Cada Época</h2>
          <button type="button" class="tr-btn-mini" id="tr-ver-todas" hidden>Ver todas</button></div>
        <div class="tr-ep-grid" id="tr-ep-grid"><div class="tr-vazio">Aguardando a primeira época...</div></div>
      </section>

      <section class="tr-log">
        <div class="tr-log-cab"><span class="tr-ico"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7l5 5-5 5M12 19h8"/></svg></span><h2>Log do Treinamento</h2>
          <div class="tr-log-acoes">
            <button type="button" id="tr-log-limpar"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/></svg>Limpar log</button>
            <label class="tr-switch"><input type="checkbox" id="tr-log-auto" checked /><i></i>Auto scroll</label>
            <button type="button" id="tr-log-exportar"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 15V3M7 8l5-5 5 5M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4"/></svg>Exportar</button>
          </div>
        </div>
        <div class="tr-log-corpo" id="log-box"><div class="tr-log-vazio">Nenhuma mensagem ainda.</div></div>
      </section>
    </div>

    <section class="card">
      <h2>Resultados</h2>
      <div id="lista-resultados" class="results-list"><p class="hint">Nenhum treinamento concluido ainda.</p></div>
    </section>
  </main>

  <div id="selecao-barra" class="selecao-barra">
    <span id="selecao-contagem" class="selecao-barra-contagem">0 selecionadas</span>
    <div class="selecao-barra-acoes">
      <button type="button" id="btn-selecao-mover" class="btn-primary">Mover</button>
      <button type="button" id="btn-selecao-cancelar" class="btn-secondary">Cancelar</button>
    </div>
  </div>

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

  <main class="content" id="view-modelos" style="display:none;">
    <div id="modelos-lista-view">
      <section class="card">
        <div class="card-header">
          <h2>Modelos</h2>
          <button type="button" id="btn-modelos-novo" class="btn-primary">+ Novo modelo</button>
        </div>
        <p class="hint">Todos os seus projetos de treinamento. Clique num deles pra ver os dados completos, imagens e resultados.</p>
        <div id="modelos-grid" class="modelos-grid"><p class="hint">Carregando modelos...</p></div>
      </section>
    </div>

    <div id="modelos-detalhe-view" style="display:none;">
      <div class="detalhe-topo">
        <div class="detalhe-titulo-row">
          <button type="button" id="btn-modelos-voltar" class="btn-secondary">&larr; Voltar</button>
          <h2 id="detalhe-nome-projeto">-</h2>
          <span id="detalhe-badge-task" class="modelo-badge detect">Deteccao</span>
        </div>
        <div style="display:flex; gap:8px;">
          <button type="button" id="btn-detalhe-renomear" class="btn-secondary">Renomear</button>
          <button type="button" id="btn-detalhe-duplicar" class="btn-secondary">Duplicar</button>
          <button type="button" id="btn-detalhe-excluir" class="btn-secondary">Excluir</button>
        </div>
      </div>

      <div class="detalhe-layout">
        <div class="detalhe-main">
          <section class="card">
            <h2>Resultados de treinamento</h2>
            <div id="detalhe-resultados-lista" class="results-list"><p class="hint">Carregando...</p></div>
          </section>
        </div>

        <div class="stat-panel">
          <div class="stat-group">
            <div class="stat-group-header">Dados do projeto</div>
            <div class="stat-grid-2" id="detalhe-dados-projeto"></div>
          </div>
          <div class="stat-group">
            <div class="stat-group-header">Classes cadastradas</div>
            <div class="stat-rows" id="detalhe-classes-lista"></div>
          </div>
        </div>
      </div>
    </div>
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
            <label id="opcao-modo-poligono"><input type="radio" name="modo-anotacao" value="poligono" /> Poligono</label>
            <label><input type="radio" name="modo-anotacao" value="varinha" /> Varinha magica</label>
          </div>
          <p id="modo-anotacao-hint" class="hint" style="display:none;">Clique nos vertices do objeto e feche clicando de novo no ponto inicial (min. 3 pontos). Esc cancela.</p>
          <div id="varinha-config-row" class="panel-block" style="display:none;">
            <span class="panel-label" title="Tamanho da area ao redor do clique onde a deteccao roda -- ajuste pra ficar um pouco maior que o objeto. Se pegar o objeto vizinho junto, diminua; se cortar pedaco do objeto, aumente.">Tamanho da selecao</span>
            <div class="size-slider-row">
              <input type="range" id="varinha-raio" min="5" max="50" step="1" value="20" />
              <span id="varinha-raio-valor" class="hint">20%</span>
            </div>
          </div>
          <h3>Caixas desenhadas</h3>
          <p class="hint" style="margin:0 0 6px;">Para corrigir uma marcacao (ex.: feita pela IA): clique nela na imagem ou na lista e escolha a classe certa (no seletor ou nos botoes de classe acima). Delete apaga a selecionada.</p>
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
        <strong id="modal-camera-titulo">Adicionar camera</strong>
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
        <span id="camera-senha-hint" class="hint" style="display:none;">Deixe em branco para manter a senha atual.</span>
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

  <div id="modal-lightbox" class="modal">
    <div class="modal-box lightbox-box">
      <div class="modal-header">
        <strong id="lightbox-titulo">Resultado</strong>
        <button id="modal-lightbox-fechar" class="icon-btn">&times;</button>
      </div>
      <div class="modal-body lightbox-body">
        <div class="lightbox-image-wrap">
          <button type="button" id="lightbox-anterior" class="lightbox-nav prev" title="Anterior">&lsaquo;</button>
          <img id="lightbox-imagem" src="" alt="Resultado" />
          <button type="button" id="lightbox-proxima" class="lightbox-nav next" title="Proxima">&rsaquo;</button>
          <span id="lightbox-contador" class="lightbox-counter"></span>
        </div>
        <div class="lightbox-side">
          <div class="stat-group">
            <div class="stat-group-header">Dados do treinamento</div>
            <div class="stat-grid-2" id="lightbox-dados-treino"></div>
          </div>
          <div class="stat-group" id="lightbox-metricas-grupo" style="display:none;">
            <div class="stat-group-header">Metricas finais</div>
            <div class="stat-grid-2" id="lightbox-metricas"></div>
          </div>
          <div id="lightbox-download-wrap"></div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const PALETTE = ['#22c55e', '#f59e0b', '#3b82f6', '#ef4444', '#a855f7', '#14b8a6', '#eab308', '#ec4899'];
    // selectedBox: indice da marcacao selecionada (-1 = nenhuma). Com uma marcacao
    // selecionada, clicar numa classe troca a classe dela.
    const state = { classes: [], activeClassIndex: 0, currentImage: null, boxes: [], drawing: false, mode: 'rect', polyPoints: [], selectedBox: -1 };
    let abaAtiva = 'pendentes';
    let projetoAtual = localStorage.getItem('treinamento_projeto_atual') || '';
    let projetoTask = 'detect';
    let modoAnotacaoPreferido = localStorage.getItem('treinamento_modo_anotacao') || 'rect';
    const DICAS_MODO = {
      rect: 'Clique e arraste para desenhar um retangulo ao redor do objeto.',
      poligono: 'Clique nos vertices do objeto e feche clicando de novo no ponto inicial (min. 3 pontos). Esc cancela.',
      varinha: 'Clique no centro do objeto -- o contorno e detectado automaticamente ao redor dele. Se pegar area demais (ou de menos), ajuste o "Tamanho da selecao" e clique de novo.',
    };

    function atualizarUiModo() {
      const modoHint = document.getElementById('modo-anotacao-hint');
      const configRow = document.getElementById('varinha-config-row');
      modoHint.textContent = DICAS_MODO[state.mode] || '';
      configRow.style.display = state.mode === 'varinha' ? 'flex' : 'none';
    }

    document.querySelectorAll('.side-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.side-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const view = btn.getAttribute('data-view');
        document.getElementById('view-treinamento').style.display = view === 'treinamento' ? 'flex' : 'none';
        document.getElementById('view-cameras').style.display = view === 'cameras' ? 'flex' : 'none';
        document.getElementById('view-testador').style.display = view === 'testador' ? 'flex' : 'none';
        document.getElementById('view-modelos').style.display = view === 'modelos' ? 'flex' : 'none';
        if (view === 'testador') carregarModelosTestador();
        if (view === 'cameras') carregarCameras();
        if (view === 'modelos') { mostrarListaModelos(); carregarModelos(); }
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
      filtroClasses = [];
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
    let editarCameraId = null;

    document.getElementById('btn-adicionar-camera').addEventListener('click', () => {
      editarCameraId = null;
      document.getElementById('modal-camera-titulo').textContent = 'Adicionar camera';
      document.getElementById('camera-senha-hint').style.display = 'none';
      ['camera-nome', 'camera-ip', 'camera-usuario', 'camera-senha', 'camera-caminho'].forEach((id) => {
        document.getElementById(id).value = '';
      });
      document.getElementById('camera-porta').value = '554';
      document.getElementById('camera-resultado').textContent = '';
      modalCamera.style.display = 'flex';
    });
    document.getElementById('modal-camera-fechar').addEventListener('click', () => { modalCamera.style.display = 'none'; });

    async function editarCamera(id) {
      try {
        const res = await fetch(`/cameras/${id}`);
        const data = await res.json();
        if (!data.ok) { alert(data.erro || 'Falha ao carregar dados dessa camera.'); return; }
        editarCameraId = id;
        document.getElementById('modal-camera-titulo').textContent = 'Editar camera';
        document.getElementById('camera-senha-hint').style.display = 'block';
        document.getElementById('camera-nome').value = data.nome;
        document.getElementById('camera-ip').value = data.ip;
        document.getElementById('camera-usuario').value = data.usuario;
        document.getElementById('camera-senha').value = '';
        document.getElementById('camera-porta').value = data.porta;
        document.getElementById('camera-caminho').value = data.caminho;
        document.getElementById('camera-resultado').textContent = '';
        modalCamera.style.display = 'flex';
      } catch (_) {
        alert('Falha ao contatar o servidor.');
      }
    }

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
      // No modo de edicao, senha em branco = mantem a senha atual (nao e
      // obrigatorio preencher de novo). No modo de adicionar, e obrigatorio.
      const senhaObrigatoria = !editarCameraId;
      if (!payload.ip || !payload.usuario || (senhaObrigatoria && !payload.senha)) {
        resultEl.textContent = 'Preencha pelo menos IP, usuario e senha.';
        resultEl.className = 'hint err';
        return;
      }
      resultEl.textContent = 'Salvando...';
      resultEl.className = 'hint';
      try {
        const url = editarCameraId ? `/cameras/${editarCameraId}/editar` : '/cameras';
        const res = await fetch(url, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (data.ok) {
          resultEl.textContent = editarCameraId ? 'Camera atualizada!' : 'Camera adicionada!';
          resultEl.className = 'hint ok';
          setTimeout(async () => { modalCamera.style.display = 'none'; await carregarCameras(); }, 400);
        } else {
          resultEl.textContent = data.erro || 'Falha ao salvar camera.';
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
                <button type="button" class="btn-secondary btn-editar-camera" data-id="${c.id}">Editar</button>
                ${c.ativa ? '' : `<button type="button" class="btn-secondary btn-remover-camera" data-id="${c.id}">Remover</button>`}
              </div>
            </div>
          `;
        }).join('');
        wrap.querySelectorAll('.btn-ativar-camera').forEach((btn) => {
          btn.addEventListener('click', () => ativarCamera(btn.getAttribute('data-id'), btn));
        });
        wrap.querySelectorAll('.btn-editar-camera').forEach((btn) => {
          btn.addEventListener('click', () => editarCamera(btn.getAttribute('data-id')));
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

    let usoClasses = [];  // [{nome, marcacoes, imagens}] na ordem das classes
    async function carregarClasses() {
      const res = await fetch(`/classes?${qp()}`);
      state.classes = await res.json();
      if (state.activeClassIndex >= state.classes.length) state.activeClassIndex = 0;
      renderClassChips();
      try {
        const r = await fetch(`/classes/uso?${qp()}`);
        usoClasses = await r.json();
        renderClassChips();
      } catch (_) {}
    }

    function renderClassChips() {
      const chip = (nome, i) => `<button type="button" class="chip-select ${i === state.activeClassIndex ? 'active' : ''}" data-idx="${i}" style="--chip-color:${PALETTE[i % PALETTE.length]}">${nome}`;
      const vazio = '<span class="hint">Nenhuma classe cadastrada ainda.</span>';
      // tela principal: classe + quantas marcacoes tem + editar / remover
      const htmlPrincipal = state.classes.length ? state.classes.map((nome, i) => {
        const u = usoClasses[i] && usoClasses[i].nome === nome ? usoClasses[i] : null;
        return `<span class="chip-classe">${chip(nome, i)}${u ? `<small title="${u.marcacoes} marcacoes em ${u.imagens} imagens">${u.marcacoes}</small>` : ''}</button>` +
          `<button type="button" class="chip-acao" data-renomear="${i}" title="Renomear a classe">&#9998;</button>` +
          `<button type="button" class="chip-acao remover" data-remover="${i}" title="Remover a classe e as marcacoes dela">&times;</button></span>`;
      }).join('') : vazio;
      // janela de anotacao: so as classes (para escolher / trocar a classe da marcacao)
      const htmlModal = state.classes.length ? state.classes.map((nome, i) => chip(nome, i) + '</button>').join('') : vazio;
      const principal = document.getElementById('lista-classes');
      const modal = document.getElementById('modal-classes');
      if (principal) principal.innerHTML = htmlPrincipal;
      if (modal) modal.innerHTML = htmlModal;
      if (principal) {
        principal.querySelectorAll('[data-renomear]').forEach((b) => b.addEventListener('click', () => renomearClasse(+b.dataset.renomear)));
        principal.querySelectorAll('[data-remover]').forEach((b) => b.addEventListener('click', () => removerClasse(+b.dataset.remover)));
      }
      document.querySelectorAll('#lista-classes .chip-select, #modal-classes .chip-select').forEach((btn) => {
        btn.addEventListener('click', () => {
          state.activeClassIndex = parseInt(btn.getAttribute('data-idx'), 10);
          // na janela de anotacao, com uma marcacao selecionada: troca a classe dela
          const noModal = btn.closest('#modal-classes') && document.getElementById('modal-anotacao').style.display === 'flex';
          if (noModal && state.boxes[state.selectedBox]) {
            state.boxes[state.selectedBox].classe = state.activeClassIndex;
            renderListaCaixas();
          }
          renderClassChips();
        });
      });
    }

    async function renomearClasse(i) {
      const atual = state.classes[i];
      const novo = (prompt(`Novo nome para a classe "${atual}":`, atual) || '').trim();
      if (!novo || novo === atual) return;
      const res = await fetch('/classes/renomear', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ projeto: projetoAtual, indice: i, nome: atual, novo_nome: novo }),
      });
      const data = await res.json();
      if (!data.ok) { alert(data.erro || 'Falha ao renomear.'); return; }
      await carregarClasses();
      carregarImagens();
    }

    async function removerClasse(i) {
      const nome = state.classes[i];
      let u = { marcacoes: 0, imagens: 0 };
      try {
        const r = await fetch(`/classes/uso?${qp()}`);
        u = (await r.json())[i] || u;
      } catch (_) {}
      const NL = String.fromCharCode(10);
      const msg = [
        `Remover a classe "${nome}"?`, '',
        u.marcacoes
          ? `Isso apaga ${u.marcacoes} marcacao(oes) dessa classe em ${u.imagens} imagem(ns) - treino, validacao e pendentes pre-anotadas.`
          : 'Nenhuma imagem tem marcacao dessa classe.',
        'As classes seguintes sao renumeradas automaticamente nas marcacoes.',
        'Antes de alterar, uma copia de seguranca das marcacoes e salva em dataset/_backup_classes.',
        'Modelos ja treinados nao mudam; o proximo treinamento ja sai sem essa classe.',
      ].join(NL);
      if (!confirm(msg)) return;
      if (u.marcacoes && prompt(`Para confirmar, digite o nome da classe: ${nome}`) !== nome) {
        alert('Nome diferente: nada foi removido.');
        return;
      }
      const res = await fetch('/classes/remover', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ projeto: projetoAtual, indice: i, nome }),
      });
      const data = await res.json();
      if (!data.ok) { alert(data.erro || 'Falha ao remover a classe.'); return; }
      // a classe ativa e o filtro usam o numero da classe: ajusta depois da renumeracao
      if (state.activeClassIndex === i) state.activeClassIndex = 0;
      else if (state.activeClassIndex > i) state.activeClassIndex -= 1;
      filtroClasses = [];
      await carregarClasses();
      await carregarImagens();
      alert([`Classe "${data.classe}" removida.`,
             `${data.marcacoes_removidas} marcacao(oes) apagada(s) em ${data.imagens_afetadas} imagem(ns).`,
             data.imagens_sem_marcacao ? `${data.imagens_sem_marcacao} imagem(ns) ficaram sem nenhuma marcacao (continuam no dataset como exemplo sem objeto).` : '',
             `Copia de seguranca: ${data.backup}`].filter(Boolean).join(NL));
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

    (function () {
      const img = document.getElementById('video-feed-img');
      const btn = document.getElementById('btn-video-toggle');
      const status = document.getElementById('video-status');
      const overlay = document.getElementById('video-pause-overlay');
      let videoParado = false;

      function atualizarUiVideo() {
        btn.textContent = videoParado ? 'Retomar video' : 'Parar video';
        status.textContent = videoParado ? 'Video pausado -- sem uso de rede.' : '';
        overlay.style.display = videoParado ? 'flex' : 'none';
      }

      btn.addEventListener('click', () => {
        videoParado = !videoParado;
        if (videoParado) {
          // Congela o ultimo frame (troca o src da propria img por uma
          // captura estatica) antes de cortar o stream -- assim nao sobra
          // o icone de imagem quebrada no lugar do video.
          let congelou = false;
          try {
            const tmp = document.createElement('canvas');
            tmp.width = img.naturalWidth || 0;
            tmp.height = img.naturalHeight || 0;
            if (tmp.width > 0 && tmp.height > 0) {
              tmp.getContext('2d').drawImage(img, 0, 0, tmp.width, tmp.height);
              img.src = tmp.toDataURL('image/jpeg', 0.85);
              congelou = true;
            }
          } catch (_) {}
          if (!congelou) img.removeAttribute('src');
          img.classList.add('video-pausado');
        } else {
          img.classList.remove('video-pausado');
          img.src = `/video_feed?t=${Date.now()}`;
        }
        atualizarUiVideo();
      });

      atualizarUiVideo();
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

    // ---- importar .zip (imagens + marcacoes da IA, exportado pelo BeefVision) ----
    const inputImportarZip = document.getElementById('input-importar-zip');
    const importarPickerTexto = document.getElementById('importar-picker-texto');
    inputImportarZip.addEventListener('change', () => {
      importarPickerTexto.textContent = inputImportarZip.files.length ? inputImportarZip.files[0].name : 'Escolher arquivo .zip';
    });
    document.getElementById('btn-importar-zip').addEventListener('click', async () => {
      if (!projetoAtual) { alert('Crie um modelo primeiro.'); return; }
      const resultEl = document.getElementById('importar-resultado');
      if (!inputImportarZip.files.length) { resultEl.textContent = 'Escolha o arquivo .zip.'; resultEl.className = 'hint err'; return; }
      const destino = document.querySelector('input[name="importar-destino"]:checked').value;
      const fd = new FormData();
      fd.append('projeto', projetoAtual);
      fd.append('destino', destino);
      fd.append('arquivo', inputImportarZip.files[0]);
      resultEl.textContent = 'Importando...';
      resultEl.className = 'hint';
      try {
        const res = await fetch('/imagens/importar', { method: 'POST', body: fd });
        const data = await res.json();
        if (data.ok) {
          const nomesDestino = { pendentes: 'Pendentes', treino: 'Treino', validacao: 'Validacao' };
          let msg = `${data.importadas} imagem(ns) importada(s) para "${nomesDestino[destino]}" com a marcacao da IA.`;
          if (data.ja_existiam) msg += ` ${data.ja_existiam} ja existiam no projeto e foram ignoradas.`;
          if (data.classes_adicionadas.length) msg += ` Classes criadas: ${data.classes_adicionadas.join(', ')}.`;
          resultEl.textContent = msg;
          resultEl.className = 'hint ok';
          inputImportarZip.value = '';
          importarPickerTexto.textContent = 'Escolher arquivo .zip';
          if (data.classes_adicionadas.length) await carregarClasses();
          carregarImagens();
        } else {
          resultEl.textContent = data.erro || 'Falha ao importar.';
          resultEl.className = 'hint err';
        }
      } catch (_) {
        resultEl.textContent = 'Falha ao contatar o servidor.';
        resultEl.className = 'hint err';
      }
    });

    let galeriaItens = [];
    let ultimosDadosImagens = { pendentes: [], treino: [], validacao: [] };
    let filtroClasses = [];

    async function carregarImagens() {
      if (!projetoAtual) return;
      const res = await fetch(`/imagens?${qp()}`);
      const data = await res.json();
      ultimosDadosImagens = data;
      document.getElementById('count-pendentes').textContent = data.pendentes.length;
      document.getElementById('count-treino').textContent = data.treino.length;
      document.getElementById('count-validacao').textContent = data.validacao.length;
      projetoTask = data.task || 'detect';
      atualizarSeletorModeloBase();
      renderFiltroClasses();
      aplicarFiltroGaleria();
    }

    function renderFiltroClasses() {
      const row = document.getElementById('filtro-classes-row');
      const wrap = document.getElementById('filtro-classes-chips');
      const btnLimpar = document.getElementById('btn-limpar-filtro-classes');
      const mostrar = abaAtiva !== 'pendentes' && state.classes.length > 0;
      row.style.display = mostrar ? 'flex' : 'none';
      if (!mostrar) return;
      btnLimpar.style.display = filtroClasses.length ? 'inline' : 'none';
      wrap.innerHTML = state.classes.map((nome, i) => `
        <button type="button" class="chip-select ${filtroClasses.includes(i) ? 'active' : ''}" data-idx="${i}" style="--chip-color:${PALETTE[i % PALETTE.length]}">${nome}</button>
      `).join('');
      wrap.querySelectorAll('.chip-select').forEach((btn) => {
        btn.addEventListener('click', () => {
          const idx = parseInt(btn.getAttribute('data-idx'), 10);
          const pos = filtroClasses.indexOf(idx);
          if (pos === -1) filtroClasses.push(idx); else filtroClasses.splice(pos, 1);
          renderFiltroClasses();
          aplicarFiltroGaleria();
        });
      });
    }

    document.getElementById('btn-limpar-filtro-classes').addEventListener('click', () => {
      filtroClasses = [];
      renderFiltroClasses();
      aplicarFiltroGaleria();
    });

    function aplicarFiltroGaleria() {
      const todosItens = ultimosDadosImagens[abaAtiva] || [];
      const emFiltro = abaAtiva !== 'pendentes' && filtroClasses.length > 0;
      const itensFiltrados = emFiltro
        ? todosItens.filter((item) => item.classes && item.classes.some((c) => filtroClasses.includes(c)))
        : todosItens;
      galeriaItens = itensFiltrados;
      const contagemEl = document.getElementById('galeria-contagem');
      if (emFiltro) {
        contagemEl.style.display = 'block';
        contagemEl.textContent = `Mostrando ${itensFiltrados.length} de ${todosItens.length} imagens`;
      } else {
        contagemEl.style.display = 'none';
      }
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

    let modoSelecao = false;
    let itensSelecionados = new Set();

    function renderGaleria(itens) {
      const wrap = document.getElementById('galeria');
      if (!itens.length) {
        wrap.innerHTML = '<p class="hint">Nenhuma imagem aqui ainda.</p>';
        return;
      }
      wrap.innerHTML = itens.map((item) => `
        <div class="thumb${modoSelecao ? ' selecionavel' : ''}${itensSelecionados.has(item.nome) ? ' selecionado' : ''}" data-nome="${item.nome}">
          <img src="/imagem?${qp({ pasta: abaAtiva, nome: item.nome })}" loading="lazy" />
          ${modoSelecao ? '<span class="thumb-select-dot"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12l5 5 9-9"/></svg></span>' : ''}
          ${item.anotado !== undefined ? `<span class="thumb-badge ${item.anotado ? 'ok' : 'warn'}">${item.anotado ? 'Anotada' : 'Sem anotacao'}</span>` : ''}
          ${item.pre_anotado ? '<span class="thumb-badge pre" title="Veio com a marcacao da IA -- abra para revisar e salvar">Marcada pela IA</span>' : ''}
          ${item.classes && item.classes.length ? `
            <div class="thumb-classes" title="Classes nesta imagem: ${item.classes.map((i) => state.classes[i] || `classe ${i}`).join(', ')}">
              ${item.classes.map((i) => `<span class="thumb-dot" style="background:${PALETTE[i % PALETTE.length]}"></span>`).join('')}
            </div>
          ` : ''}
          ${modoSelecao ? '' : `
          <div class="thumb-actions">
            <button class="thumb-btn" data-acao="anotar">Anotar</button>
            <button class="thumb-btn" data-acao="mover">${abaAtiva === 'treino' ? 'Mover p/ validacao' : 'Mover p/ treino'}</button>
            <button class="thumb-btn danger" data-acao="excluir">Excluir</button>
          </div>
          `}
        </div>
      `).join('');

      if (modoSelecao) {
        wrap.querySelectorAll('.thumb').forEach((el) => {
          el.addEventListener('click', () => alternarSelecaoItem(el.getAttribute('data-nome')));
        });
      } else {
        wrap.querySelectorAll('.thumb').forEach((el) => {
          const nome = el.getAttribute('data-nome');
          el.querySelector('[data-acao="anotar"]').addEventListener('click', () => abrirAnotacao(abaAtiva, nome));
          const moverBtn = el.querySelector('[data-acao="mover"]');
          if (moverBtn) moverBtn.addEventListener('click', () => moverImagem(nome));
          el.querySelector('[data-acao="excluir"]').addEventListener('click', () => excluirImagem(nome));
        });
      }
    }

    function alternarSelecaoItem(nome) {
      if (itensSelecionados.has(nome)) itensSelecionados.delete(nome);
      else itensSelecionados.add(nome);
      renderGaleria(galeriaItens);
      atualizarBarraSelecao();
    }

    function atualizarBarraSelecao() {
      const barra = document.getElementById('selecao-barra');
      const n = itensSelecionados.size;
      barra.classList.toggle('ativa', modoSelecao);
      document.getElementById('selecao-contagem').textContent = `${n} selecionada${n === 1 ? '' : 's'}`;
      const btnMover = document.getElementById('btn-selecao-mover');
      btnMover.disabled = n === 0;
      btnMover.textContent = abaAtiva === 'treino' ? 'Mover p/ validacao' : 'Mover p/ treino';
    }

    function sairModoSelecao() {
      modoSelecao = false;
      itensSelecionados.clear();
      document.getElementById('btn-modo-selecao').textContent = 'Selecionar';
      renderGaleria(galeriaItens);
      atualizarBarraSelecao();
    }

    document.getElementById('btn-modo-selecao').addEventListener('click', () => {
      modoSelecao = !modoSelecao;
      itensSelecionados.clear();
      document.getElementById('btn-modo-selecao').textContent = modoSelecao ? 'Cancelar selecao' : 'Selecionar';
      renderGaleria(galeriaItens);
      atualizarBarraSelecao();
    });

    document.getElementById('btn-selecao-cancelar').addEventListener('click', sairModoSelecao);

    document.getElementById('btn-selecao-mover').addEventListener('click', async () => {
      if (!itensSelecionados.size) return;
      const destino = abaAtiva === 'treino' ? 'validacao' : 'treino';
      const nomes = [...itensSelecionados];
      const btnMover = document.getElementById('btn-selecao-mover');
      btnMover.disabled = true;
      btnMover.textContent = 'Movendo...';
      await Promise.all(nomes.map((nome) => fetch('/imagens/mover', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ projeto: projetoAtual, nome, origem: abaAtiva, destino }),
      })));
      sairModoSelecao();
      await carregarImagens();
    });

    document.querySelectorAll('.tab-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        abaAtiva = btn.getAttribute('data-tab');
        sairModoSelecao();
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
      state.selectedBox = -1;
      document.getElementById('modal-titulo').textContent = `Anotar: ${nome}`;
      document.getElementById('anotacao-resultado').textContent = '';
      const modoRow = document.getElementById('modo-anotacao-row');
      const modoHint = document.getElementById('modo-anotacao-hint');
      const opcaoPoligono = document.getElementById('opcao-modo-poligono');
      // O modo (retangulo/varinha magica) vale pra qualquer tipo de
      // projeto -- so o poligono manual e exclusivo de projetos de
      // segmentacao (num projeto de deteccao simples nao faz sentido
      // desenhar um poligono, ja que so se guarda a caixa retangular).
      modoRow.style.display = 'flex';
      modoHint.style.display = 'block';
      opcaoPoligono.style.display = projetoTask === 'segment' ? '' : 'none';
      state.mode = modoAnotacaoPreferido;
      if (state.mode === 'poligono' && projetoTask !== 'segment') state.mode = 'rect';
      document.querySelector(`input[name="modo-anotacao"][value="${state.mode}"]`).checked = true;
      atualizarUiModo();
      renderClassChips();
      renderListaCaixas();
      atualizarNavegacaoAnotacao();

      // pendentes tambem: pode ter vindo com a marcacao da IA (importada do BeefVision)
      try {
        const res = await fetch(`/anotacao?${qp({ pasta, nome })}`);
        if (res.ok) {
          const data = await res.json();
          if (state.currentImage && state.currentImage.nome === nome) state.boxes = data.caixas || [];
        }
      } catch (_) {}

      const img = new Image();
      img.onload = () => {
        imagemCarregada = img;
        // Altura da linha e fixa via CSS (nao depende do tamanho da
        // imagem) -- limitando o canvas a essa altura, as setas de
        // navegacao ficam sempre na mesma posicao entre uma foto e outra,
        // independente da proporcao de cada imagem.
        const navRow = document.querySelector('.canvas-nav-row');
        const maxW = Math.min(760, window.innerWidth - 430);
        const maxH = navRow ? navRow.clientHeight : 520;
        const scale = Math.min(1, maxW / img.naturalWidth, (maxH || 520) / img.naturalHeight);
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
      state.boxes.forEach((box, idx) => {
        const cor = PALETTE[box.classe % PALETTE.length];
        const selecionada = idx === state.selectedBox;
        let x, y;
        if (box.pontos) {
          const pontosCanvas = box.pontos.map(([px, py]) => [px * canvas.width, py * canvas.height]);
          if (selecionada) {
            ctx.beginPath();
            pontosCanvas.forEach(([px, py], i) => { if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); });
            ctx.closePath();
            ctx.fillStyle = cor + '55';
            ctx.fill();
          }
          desenharPoligono(pontosCanvas, cor, true);
          if (selecionada) { ctx.lineWidth = 4; ctx.stroke(); }
          x = Math.min(...pontosCanvas.map((p) => p[0]));
          y = Math.min(...pontosCanvas.map((p) => p[1]));
        } else {
          x = box.x * canvas.width; y = box.y * canvas.height;
          const w = box.w * canvas.width, h = box.h * canvas.height;
          if (selecionada) { ctx.fillStyle = cor + '55'; ctx.fillRect(x, y, w, h); }
          ctx.strokeStyle = cor; ctx.lineWidth = selecionada ? 4 : 2;
          ctx.strokeRect(x, y, w, h);
        }
        ctx.lineWidth = 2;
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
        state.selectedBox = -1;
        renderListaCaixas();
      } else {
        // clique sem arrastar: seleciona a marcacao que estiver embaixo do mouse
        selecionarCaixaEm(state.startX / canvas.width, state.startY / canvas.height);
      }
    });

    function pontoNoPoligono(x, y, pts) {
      let dentro = false;
      for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
        const [xi, yi] = pts[i], [xj, yj] = pts[j];
        if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) dentro = !dentro;
      }
      return dentro;
    }

    // entre as marcacoes que contem o ponto, escolhe a menor (a mais "por cima")
    function selecionarCaixaEm(x, y) {
      let melhor = -1, menorArea = Infinity;
      state.boxes.forEach((box, i) => {
        let dentro, area;
        if (box.pontos) {
          dentro = pontoNoPoligono(x, y, box.pontos);
          const xs = box.pontos.map((p) => p[0]), ys = box.pontos.map((p) => p[1]);
          area = (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys));
        } else {
          dentro = x >= box.x && x <= box.x + box.w && y >= box.y && y <= box.y + box.h;
          area = box.w * box.h;
        }
        if (dentro && area < menorArea) { melhor = i; menorArea = area; }
      });
      selecionarCaixa(melhor);
    }

    function selecionarCaixa(i) {
      state.selectedBox = i;
      if (state.boxes[i]) state.activeClassIndex = state.boxes[i].classe;
      renderClassChips();
      renderListaCaixas();
    }

    canvas.addEventListener('click', async (ev) => {
      if (state.mode === 'poligono') {
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
        return;
      }

      if (state.mode === 'varinha') {
        if (!state.classes.length) { alert('Cadastre uma classe antes de selecionar.'); return; }
        if (!state.currentImage || canvas.width === 0) return;
        const [x, y] = posicaoNoCanvas(ev);
        const xNorm = x / canvas.width;
        const yNorm = y / canvas.height;
        const raioPercentual = document.getElementById('varinha-raio').value;
        const resultEl = document.getElementById('anotacao-resultado');
        canvas.style.cursor = 'progress';
        resultEl.textContent = 'Detectando contorno...';
        resultEl.className = 'hint';
        try {
          const res = await fetch('/anotar/varinha_magica', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              projeto: projetoAtual, pasta: state.currentImage.pasta, nome: state.currentImage.nome,
              x: xNorm, y: yNorm, raio_percentual: raioPercentual,
            }),
          });
          const data = await res.json();
          if (data.ok) {
            if (projetoTask === 'segment') {
              state.boxes.push({ classe: state.activeClassIndex, pontos: data.pontos });
            } else {
              // Projeto de deteccao simples (caixa): a varinha ainda detecta
              // o contorno do objeto automaticamente, mas o que se guarda e
              // so a caixa retangular ao redor dele (formato que esse tipo
              // de projeto usa), nao o poligono.
              const xs = data.pontos.map((p) => p[0]);
              const ys = data.pontos.map((p) => p[1]);
              const minX = Math.min(...xs), maxX = Math.max(...xs);
              const minY = Math.min(...ys), maxY = Math.max(...ys);
              state.boxes.push({ classe: state.activeClassIndex, x: minX, y: minY, w: maxX - minX, h: maxY - minY });
            }
            resultEl.textContent = '';
            renderListaCaixas();
          } else {
            resultEl.textContent = data.erro || 'Nao foi possivel detectar o contorno nesse ponto.';
            resultEl.className = 'hint err';
          }
        } catch (_) {
          resultEl.textContent = 'Falha ao contatar o servidor.';
          resultEl.className = 'hint err';
        } finally {
          canvas.style.cursor = 'crosshair';
        }
      }
    });

    document.querySelectorAll('input[name="modo-anotacao"]').forEach((r) => {
      r.addEventListener('change', (ev) => {
        state.mode = ev.target.value;
        modoAnotacaoPreferido = state.mode;
        localStorage.setItem('treinamento_modo_anotacao', modoAnotacaoPreferido);
        state.polyPoints = [];
        state.drawing = false;
        atualizarUiModo();
        desenharCanvas();
      });
    });

    (function () {
      const slider = document.getElementById('varinha-raio');
      const valorEl = document.getElementById('varinha-raio-valor');
      let salvo = null;
      try { salvo = localStorage.getItem('treinamento_varinha_raio'); } catch (_) {}
      if (salvo && !Number.isNaN(parseInt(salvo, 10))) slider.value = salvo;
      valorEl.textContent = `${slider.value}%`;
      slider.addEventListener('input', () => {
        valorEl.textContent = `${slider.value}%`;
        try { localStorage.setItem('treinamento_varinha_raio', slider.value); } catch (_) {}
      });
    })();

    window.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape' && state.mode === 'poligono' && state.polyPoints.length) {
        state.polyPoints = [];
        desenharCanvas();
        return;
      }
      if (modal.style.display !== 'flex' || /INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) return;
      // Delete: apaga a marcacao selecionada; Esc: tira a selecao;
      // setas do teclado: troca de foto (igual aos botoes de navegacao)
      if ((ev.key === 'Delete' || ev.key === 'Backspace') && state.boxes[state.selectedBox]) {
        ev.preventDefault();
        removerCaixa(state.selectedBox);
      } else if (ev.key === 'Escape' && state.selectedBox >= 0) {
        selecionarCaixa(-1);
      } else if (ev.key === 'ArrowLeft') {
        ev.preventDefault();
        navegarAnotacao(-1);
      } else if (ev.key === 'ArrowRight') {
        ev.preventDefault();
        navegarAnotacao(1);
      }
    });

    function removerCaixa(i) {
      state.boxes.splice(i, 1);
      if (state.selectedBox === i) state.selectedBox = -1;
      else if (state.selectedBox > i) state.selectedBox -= 1;
      renderListaCaixas();
    }

    function renderListaCaixas() {
      const ul = document.getElementById('lista-caixas');
      if (!state.boxes.length) {
        ul.innerHTML = '<li class="hint">Nenhuma caixa desenhada.</li>';
      } else {
        // cada marcacao tem um seletor para trocar a classe (ex.: corrigir o que a IA marcou)
        const opcoes = (atual) => state.classes.map((nome, k) =>
          `<option value="${k}" ${k === atual ? 'selected' : ''}>${nome}</option>`).join('')
          + (atual >= state.classes.length ? `<option value="${atual}" selected>? (classe ${atual})</option>` : '');
        ul.innerHTML = state.boxes.map((box, i) => `
          <li data-sel="${i}" class="${i === state.selectedBox ? 'sel' : ''}" title="Clique para selecionar esta marcacao">
            <span style="color:${PALETTE[box.classe % PALETTE.length]}">&#9632;</span>
            <select data-classe="${i}" title="Trocar a classe desta marcacao">${opcoes(box.classe)}</select>
            <button type="button" data-idx="${i}" class="link-btn-sm">remover</button>
          </li>
        `).join('');
        ul.querySelectorAll('button[data-idx]').forEach((btn) => {
          btn.addEventListener('click', (ev) => {
            ev.stopPropagation();
            removerCaixa(parseInt(btn.getAttribute('data-idx'), 10));
          });
        });
        ul.querySelectorAll('select[data-classe]').forEach((sel) => {
          sel.addEventListener('click', (ev) => ev.stopPropagation());
          sel.addEventListener('change', () => {
            const i = parseInt(sel.getAttribute('data-classe'), 10);
            state.boxes[i].classe = parseInt(sel.value, 10);
            state.selectedBox = i;
            state.activeClassIndex = state.boxes[i].classe;
            renderClassChips();
            renderListaCaixas();
          });
        });
        ul.querySelectorAll('li[data-sel]').forEach((li) => {
          li.addEventListener('click', () => {
            const i = parseInt(li.getAttribute('data-sel'), 10);
            selecionarCaixa(i === state.selectedBox ? -1 : i);
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
          // Pendentes: a imagem salva sai da lista (foi pra treino/validacao),
          // entao em vez de fechar o popup e te obrigar a clicar na proxima,
          // abre direto a que ficou na mesma posicao dela na lista atualizada.
          const idxSalva = galeriaItens.findIndex((it) => it.nome === state.currentImage.nome);
          await carregarImagens();
          if (galeriaItens.length) {
            const proximoIdx = Math.min(idxSalva < 0 ? 0 : idxSalva, galeriaItens.length - 1);
            abrirAnotacao('pendentes', galeriaItens[proximoIdx].nome);
          } else {
            resultEl.textContent = 'Salvo! Nao ha mais pendentes.';
            resultEl.className = 'hint ok';
            setTimeout(() => { modal.style.display = 'none'; }, 800);
          }
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
      document.getElementById('treino-progresso').style.display = 'grid';
      btn.disabled = true;
      clearInterval(treinoPollTimer);
      treinoPollTimer = setInterval(atualizarStatusTreino, 2000);
      atualizarStatusTreino();
    });

    // ---- tempo das epocas ----
    // 72 -> "1m 12s"; 3780 -> "1h 03m"; 9.4 -> "9.4s"
    function fmtDuracao(seg) {
      if (seg == null || isNaN(seg)) return '-';
      if (seg < 9.95) return `${seg.toFixed(1)}s`;
      if (Math.round(seg) < 60) return `${Math.round(seg)}s`;  // arredonda antes: 59.6 vira "1m 00s"
      let m = Math.floor(Math.round(seg) / 60), s = Math.round(seg) % 60;
      if (m < 60) return `${m}m ${String(s).padStart(2, '0')}s`;
      const h = Math.floor(m / 60); m = m % 60;
      return `${h}h ${String(m).padStart(2, '0')}m`;
    }
    // resumo (total, media, mais rapida/lenta, quanto falta) + o tempo de cada epoca
    function temposHtml(t, comLista) {
      if (!t || !t.epocas || !t.epocas.length) return '';
      const pills = [
        ['Ultima epoca', t.epocas[t.epocas.length - 1]], ['Media por epoca', t.media], ['Tempo total', t.total],
        ['Mais rapida', t.mais_rapida], ['Mais lenta', t.mais_lenta],
      ];
      if (t.restante != null) pills.push(['Falta aprox.', t.restante]);
      const limiteLenta = t.media * 1.5;  // destaca epocas bem mais lentas que a media
      const lista = comLista ? `
        <div class="tempos-titulo">Tempo de cada epoca</div>
        <div class="tempos-lista">${t.epocas.map((s, i) =>
          `<span class="${s > limiteLenta ? 'lenta' : ''}" title="${s > limiteLenta ? 'Bem mais lenta que a media' : ''}">Epoca ${i + 1}<b>${fmtDuracao(s)}</b></span>`).join('')}</div>` : '';
      return `<div class="tempos-resumo">${pills.map(([k, v]) => `<span class="metric-pill">${k}: <b>${fmtDuracao(v)}</b></span>`).join('')}</div>${lista}`;
    }

    // ================= painel do treinamento =================
    // cada metrica: rotulo, cor (paleta validada para daltonismo) e icone
    const TR_ICONES = {
      alvo: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.5" fill="currentColor"/></svg>',
      lista: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="5" y="4" width="14" height="17" rx="2"/><path d="M9 9h6M9 13h6M9 17h3"/></svg>',
      linha: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17l6-6 4 4 8-8"/></svg>',
      caixa: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M4 7.5l8 4.5 8-4.5M12 12v9"/></svg>',
      camadas: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/></svg>',
      etiqueta: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><path d="M3 12V4h8l10 10-8 8z"/><circle cx="7.5" cy="8.5" r="1.5" fill="currentColor"/></svg>',
      nos: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="6" cy="12" r="2.5"/><circle cx="18" cy="6" r="2.5"/><circle cx="18" cy="18" r="2.5"/><path d="M8.2 11l7.6-4M8.2 13l7.6 4"/></svg>',
      ponto: '<svg viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="4"/></svg>',
    };
    const TR_SERIES = {
      metricas: [
        ['precisao', 'Precisão', '#2563eb', 'alvo'], ['recall', 'Recall', '#16a34a', 'lista'],
        ['mAP50', 'mAP50', '#7c3aed', 'linha'], ['mAP50-95', 'mAP50-95', '#ea580c', 'linha'],
      ],
      perdas: [
        ['box_loss', 'Box Loss', '#dc2626', 'caixa'], ['seg_loss', 'Seg Loss', '#a16207', 'camadas'],
        ['cls_loss', 'Cls Loss', '#0891b2', 'etiqueta'], ['dfl_loss', 'DFL Loss', '#db2777', 'nos'],
      ],
    };
    const trEstado = { modo: 'metricas', verTodas: false, logLimpoAte: null, assinaturaGraf: '', assinaturaLog: '', ultimo: null };
    const trEsc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    const trDef = (chave) => [...TR_SERIES.metricas, ...TR_SERIES.perdas].find((d) => d[0] === chave)
      || [chave, chave.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase()), '#6b7280', 'ponto'];

    function trTiles(metrics) {
      const chaves = Object.keys(metrics || {});
      if (!chaves.length) return '<div class="tr-vazio">Aguardando a primeira época...</div>';
      // metricas primeiro, perdas depois, na ordem da paleta
      const ordem = [...TR_SERIES.metricas, ...TR_SERIES.perdas].map((d) => d[0]);
      chaves.sort((a, b) => (ordem.indexOf(a) < 0 ? 99 : ordem.indexOf(a)) - (ordem.indexOf(b) < 0 ? 99 : ordem.indexOf(b)));
      return chaves.map((k) => {
        const [, rotulo, cor, ico] = trDef(k);
        const v = Number(metrics[k]);
        return `<div class="tr-tile" style="--c:${cor}"><span class="tr-tile-ico">${TR_ICONES[ico]}</span>` +
          `<div><span>${trEsc(rotulo)}</span><b>${isNaN(v) ? '-' : v.toFixed(4)}</b></div></div>`;
      }).join('');
    }

    // grafico de linhas por epoca: "Metricas" (0 a 1) ou "Perdas" (escala propria) - um eixo por vez
    function trGrafico(historico, total) {
      const area = document.getElementById('tr-graf');
      const series = TR_SERIES[trEstado.modo].filter((d) => historico.some((h) => h.metricas[d[0]] != null));
      document.getElementById('tr-legenda').innerHTML = series.map(([, rot, cor]) =>
        `<span><i style="background:${cor}"></i>${trEsc(rot)}</span>`).join('');
      const largura = Math.max(320, area.clientWidth || 600);
      const assinatura = `${trEstado.modo}|${historico.length}|${total}|${largura}`;
      if (assinatura === trEstado.assinaturaGraf) return;
      trEstado.assinaturaGraf = assinatura;
      if (!historico.length || !series.length) {
        area.innerHTML = '<div class="tr-vazio">O gráfico aparece depois da primeira época.</div>';
        return;
      }
      const W = largura, H = 270, ml = 40, mr = 14, mt = 10, mb = 30;
      const xMax = Math.max(total || 1, historico[historico.length - 1].epoca, 2);
      let yMax = 1;
      if (trEstado.modo === 'perdas') {
        const maior = Math.max(...historico.flatMap((h) => series.map((d) => Number(h.metricas[d[0]]) || 0)));
        const passo = maior > 2 ? 1 : maior > 1 ? 0.5 : 0.2;
        yMax = Math.max(passo, Math.ceil(maior / passo) * passo);
      }
      const X = (e) => ml + ((e - 1) / Math.max(1, xMax - 1)) * (W - ml - mr);
      const Y = (v) => mt + (1 - v / yMax) * (H - mt - mb);
      let grade = '';
      for (let k = 0; k <= 5; k++) {
        const v = (yMax * k) / 5, y = Y(v);
        grade += `<line x1="${ml}" x2="${W - mr}" y1="${y}" y2="${y}" stroke="#ececf0" stroke-width="1"/>` +
          `<text class="eixo" x="${ml - 8}" y="${y + 4}" text-anchor="end">${v.toFixed(yMax < 2 ? 1 : 0)}</text>`;
      }
      const passoX = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500].find((p) => xMax / p <= 10) || 1000;
      for (let e = passoX; e <= xMax; e += passoX)
        grade += `<text class="eixo" x="${X(e)}" y="${H - 9}" text-anchor="middle">${e}</text>`;
      grade += `<text class="eixo" x="${X(1)}" y="${H - 9}" text-anchor="middle">1</text>`;
      const linhas = series.map(([chave, , cor]) => {
        const pts = historico.filter((h) => h.metricas[chave] != null).map((h) => `${X(h.epoca).toFixed(1)},${Y(Number(h.metricas[chave])).toFixed(1)}`);
        const ult = pts[pts.length - 1].split(',');
        return `<polyline points="${pts.join(' ')}" fill="none" stroke="${cor}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>` +
          `<circle cx="${ult[0]}" cy="${ult[1]}" r="4" fill="${cor}" stroke="#fff" stroke-width="2"/>`;
      }).join('');
      area.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="Evolução por época">
          ${grade}${linhas}
          <line id="tr-guia" x1="0" x2="0" y1="${mt}" y2="${H - mb}" stroke="#9ca3af" stroke-width="1" visibility="hidden"/>
          <g id="tr-pontos"></g>
          <rect id="tr-alvo" x="${ml}" y="${mt}" width="${W - ml - mr}" height="${H - mt - mb}" fill="transparent"/>
        </svg><div class="tr-dica-graf" id="tr-dica" hidden></div>`;
      // passar o mouse: linha guia + valores da epoca mais proxima
      const alvo = document.getElementById('tr-alvo');
      alvo.addEventListener('mousemove', (ev) => {
        const caixa = alvo.ownerSVGElement.getBoundingClientRect();
        const xs = (ev.clientX - caixa.left) * (W / caixa.width);
        const e = Math.round(1 + ((xs - ml) / (W - ml - mr)) * (xMax - 1));
        const h = historico.reduce((m, a) => (Math.abs(a.epoca - e) < Math.abs(m.epoca - e) ? a : m), historico[0]);
        const guia = document.getElementById('tr-guia');
        guia.setAttribute('x1', X(h.epoca)); guia.setAttribute('x2', X(h.epoca)); guia.setAttribute('visibility', 'visible');
        document.getElementById('tr-pontos').innerHTML = series.filter((d) => h.metricas[d[0]] != null).map(([chave, , cor]) =>
          `<circle cx="${X(h.epoca)}" cy="${Y(Number(h.metricas[chave]))}" r="4.5" fill="${cor}" stroke="#fff" stroke-width="2"/>`).join('');
        const dica = document.getElementById('tr-dica');
        dica.innerHTML = `<b>Época ${h.epoca}</b><br>` + series.filter((d) => h.metricas[d[0]] != null).map(([chave, rot, cor]) =>
          `<i style="background:${cor}"></i>${trEsc(rot)}: <b>${Number(h.metricas[chave]).toFixed(4)}</b>`).join('<br>');
        dica.hidden = false;
        const px = (X(h.epoca) / W) * caixa.width;
        dica.style.left = `${Math.min(Math.max(px + 14, 0), caixa.width - dica.offsetWidth - 4)}px`;
        dica.style.top = '8px';
        if (px + 14 + dica.offsetWidth > caixa.width) dica.style.left = `${Math.max(0, px - dica.offsetWidth - 14)}px`;
      });
      alvo.addEventListener('mouseleave', () => {
        document.getElementById('tr-guia').setAttribute('visibility', 'hidden');
        document.getElementById('tr-pontos').innerHTML = '';
        document.getElementById('tr-dica').hidden = true;
      });
    }

    function trTempos(t) {
      const grade = document.getElementById('tr-ep-grid');
      const botao = document.getElementById('tr-ver-todas');
      const epocas = (t && t.epocas) || [];
      if (!epocas.length) { grade.innerHTML = '<div class="tr-vazio">Aguardando a primeira época...</div>'; botao.hidden = true; return; }
      const LIMITE = 24;
      botao.hidden = epocas.length <= LIMITE;
      botao.textContent = trEstado.verTodas ? 'Ver menos' : `Ver todas (${epocas.length})`;
      const inicio = trEstado.verTodas ? 0 : Math.max(0, epocas.length - LIMITE);
      const lenta = t.media * 1.5;  // destaca epocas bem mais lentas que a media
      const rolagem = grade.scrollTop;
      grade.classList.toggle('todas', trEstado.verTodas);
      grade.innerHTML = epocas.slice(inicio).map((seg, k) =>
        `<div class="tr-ep ${seg > lenta ? 'lenta' : ''}" title="${seg > lenta ? 'Bem mais lenta que a média' : ''}">` +
        `<span>Ep ${inicio + k + 1}</span><b>${fmtDuracao(seg)}</b></div>`).join('');
      grade.scrollTop = rolagem;
    }

    // log: [hora] NIVEL mensagem
    function trLog(linhas) {
      const corpo = document.getElementById('log-box');
      let mostrar = linhas;
      if (trEstado.logLimpoAte != null) {
        const pos = linhas.lastIndexOf(trEstado.logLimpoAte);
        mostrar = pos >= 0 ? linhas.slice(pos + 1) : linhas;
      }
      const assinatura = mostrar.length + '|' + (mostrar[mostrar.length - 1] || '');
      if (assinatura === trEstado.assinaturaLog) return;
      trEstado.assinaturaLog = assinatura;
      corpo.innerHTML = mostrar.length ? mostrar.map((l) => {
        const temHora = l[0] === '[' && l[9] === ']';  // "[HH:MM:SS] mensagem"
        const hora = temHora ? l.slice(1, 9) : '', msg = temHora ? l.slice(11) : l;
        const nivel = /erro|falha|error/i.test(msg) ? 'erro' : /aviso|warning/i.test(msg) ? 'aviso' : 'info';
        const texto = trEsc(msg).replace(/^(Epoca [0-9]+.[0-9]+)/, '<span class="ep">$1</span>');
        return `<div>${hora ? `<span class="h">[${hora}]</span>` : ''}<span class="n ${nivel}">${nivel.toUpperCase()}</span>${texto}</div>`;
      }).join('') : '<div class="tr-log-vazio">Nenhuma mensagem ainda.</div>';
      if (document.getElementById('tr-log-auto').checked) corpo.scrollTop = corpo.scrollHeight;
    }

    function trPainel(s) {
      trEstado.ultimo = s;
      const t = s.tempos || {};
      const pct = s.epochs_total > 0 ? Math.round((s.epoch / s.epochs_total) * 100) : 0;
      const concluido = s.status === 'concluido', erro = s.status === 'erro';
      document.getElementById('tr-device').textContent = s.device ? `Rodando em: ${s.device}` : '';
      document.getElementById('tr-epoca-txt').textContent = s.epochs_total
        ? `Época ${s.epoch} de ${s.epochs_total}` : 'Preparando o treinamento...';
      document.getElementById('tr-pct').textContent = `${concluido ? 100 : pct}%`;
      const barra = document.getElementById('tr-barra');
      barra.classList.toggle('ok', concluido); barra.classList.toggle('erro', erro);
      document.getElementById('tr-barra-fill').style.width = `${concluido ? 100 : pct}%`;
      const msg = document.getElementById('tr-msg');
      msg.className = 'tr-msg' + (concluido ? ' ok' : erro ? ' erro' : '');
      msg.textContent = concluido ? `Treinamento concluído${s.finished_at ? ` em ${s.finished_at}` : ''}.`
        : erro ? `Erro: ${s.error}` : '';
      document.getElementById('tr-st-total').textContent = fmtDuracao(s.decorrido);
      document.getElementById('tr-st-atual').textContent = s.running ? fmtDuracao(s.epoca_atual_seg) : '-';
      document.getElementById('tr-st-media').textContent = fmtDuracao(t.media);
      document.getElementById('tr-st-restante').textContent = s.running ? fmtDuracao(t.restante) : '-';
      document.getElementById('tr-metr-epoca').textContent = s.epoch ? `(Época ${s.epoch}/${s.epochs_total})` : '';
      document.getElementById('tr-tiles').innerHTML = trTiles(s.metrics);
      trGrafico(s.historico || [], s.epochs_total);
      trTempos(t);
      trLog(s.log || []);
    }

    document.querySelectorAll('#tr-graf-modo button').forEach((b) => b.addEventListener('click', () => {
      trEstado.modo = b.dataset.modo;
      document.querySelectorAll('#tr-graf-modo button').forEach((x) => x.classList.toggle('ativo', x === b));
      if (trEstado.ultimo) trGrafico(trEstado.ultimo.historico || [], trEstado.ultimo.epochs_total);
    }));
    document.getElementById('tr-ver-todas').addEventListener('click', () => {
      trEstado.verTodas = !trEstado.verTodas;
      if (trEstado.ultimo) trTempos(trEstado.ultimo.tempos || {});
    });
    document.getElementById('tr-log-limpar').addEventListener('click', () => {
      const linhas = (trEstado.ultimo && trEstado.ultimo.log) || [];
      trEstado.logLimpoAte = linhas[linhas.length - 1] || null;
      trEstado.assinaturaLog = '';
      trLog(linhas);
    });
    document.getElementById('tr-log-exportar').addEventListener('click', () => {
      const linhas = (trEstado.ultimo && trEstado.ultimo.log) || [];
      if (!linhas.length) { alert('O log está vazio.'); return; }
      const blob = new Blob([linhas.join(String.fromCharCode(13, 10))], { type: 'text/plain;charset=utf-8' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `log_treinamento_${(trEstado.ultimo.projeto || 'modelo').replace(/[^a-zA-Z0-9_-]/g, '_')}.txt`;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 30000);
    });
    window.addEventListener('resize', () => { if (trEstado.ultimo) trGrafico(trEstado.ultimo.historico || [], trEstado.ultimo.epochs_total); });

    async function atualizarStatusTreino() {
      if (!projetoAtual) return;
      const res = await fetch('/treinamento/status');
      const s = await res.json();
      const btn = document.getElementById('btn-iniciar-treino');
      const ehEsteProjeto = s.projeto === projetoAtual;
      const painel = document.getElementById('treino-progresso');

      if (s.status !== 'ocioso' && ehEsteProjeto) {
        if (painel.style.display === 'none') trEstado.assinaturaGraf = '';  // redesenha com a largura certa
        painel.style.display = 'grid';
        trPainel(s);
      } else if (s.running) {
        painel.style.display = 'none';
        document.getElementById('ambiente-info').textContent =
          `Há um treinamento em andamento no modelo "${s.projeto}". Aguarde terminar para treinar este.`;
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
          ${r.tempos && r.tempos.total ? `<div class="hint">${r.tempos.epocas.length} epocas em ${fmtDuracao(r.tempos.total)} (media ${fmtDuracao(r.tempos.media)} por epoca)</div>` : ''}
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

    // ==================== pagina Modelos ====================

    function mostrarListaModelos() {
      document.getElementById('modelos-detalhe-view').style.display = 'none';
      document.getElementById('modelos-lista-view').style.display = 'block';
    }
    document.getElementById('btn-modelos-voltar').addEventListener('click', mostrarListaModelos);

    document.getElementById('btn-modelos-novo').addEventListener('click', () => {
      document.getElementById('btn-novo-projeto').click();
    });

    async function carregarModelos() {
      const wrap = document.getElementById('modelos-grid');
      try {
        const res = await fetch('/modelos');
        const lista = await res.json();
        if (!lista.length) {
          wrap.innerHTML = '<p class="hint">Nenhum modelo cadastrado ainda. Clique em "+ Novo modelo".</p>';
          return;
        }
        wrap.innerHTML = lista.map((m) => `
          <div class="modelo-card" data-nome="${m.nome}">
            <div class="modelo-card-header">
              <strong>${m.nome}</strong>
              <span class="modelo-badge ${m.task}">${m.task === 'segment' ? 'Segmentacao' : 'Deteccao'}</span>
            </div>
            <div class="modelo-stats-mini">
              <div class="modelo-stat-mini"><strong>${m.total_treino}</strong><span>Treino</span></div>
              <div class="modelo-stat-mini"><strong>${m.total_validacao}</strong><span>Validacao</span></div>
              <div class="modelo-stat-mini"><strong>${m.total_classes}</strong><span>Classes</span></div>
            </div>
            <div class="modelo-card-footer">
              <span class="hint">${m.total_treinamentos} treinamento(s)${m.tem_modelo ? ' &middot; modelo pronto' : ''}<br />${m.ultima_modificacao}</span>
              <div class="modelo-card-actions">
                <button type="button" data-acao="renomear" title="Renomear">&#9998;</button>
                <button type="button" data-acao="duplicar" title="Duplicar">&#10697;</button>
                <button type="button" data-acao="excluir" class="danger" title="Excluir">&#128465;</button>
              </div>
            </div>
          </div>
        `).join('');
        wrap.querySelectorAll('.modelo-card').forEach((card) => {
          const nome = card.getAttribute('data-nome');
          card.addEventListener('click', () => abrirModeloDetalhe(nome));
          card.querySelector('[data-acao="renomear"]').addEventListener('click', async (ev) => {
            ev.stopPropagation();
            if (await renomearModelo(nome)) carregarModelos();
          });
          card.querySelector('[data-acao="duplicar"]').addEventListener('click', async (ev) => {
            ev.stopPropagation();
            if (await duplicarModelo(nome)) carregarModelos();
          });
          card.querySelector('[data-acao="excluir"]').addEventListener('click', async (ev) => {
            ev.stopPropagation();
            if (await excluirModelo(nome)) carregarModelos();
          });
        });
      } catch (_) {
        wrap.innerHTML = '<p class="hint err">Falha ao carregar os modelos.</p>';
      }
    }

    async function renomearModelo(nome) {
      const novoNome = prompt('Novo nome do modelo:', nome);
      if (!novoNome || !novoNome.trim() || novoNome.trim() === nome) return false;
      const res = await fetch(`/modelos/${encodeURIComponent(nome)}/renomear`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ novo_nome: novoNome.trim() }),
      });
      const data = await res.json();
      if (!data.ok) { alert(data.erro || 'Falha ao renomear.'); return false; }
      if (projetoAtual === nome) {
        projetoAtual = data.projeto;
        localStorage.setItem('treinamento_projeto_atual', projetoAtual);
      }
      await carregarProjetos();
      return true;
    }

    async function duplicarModelo(nome) {
      const novoNome = prompt('Nome do modelo duplicado:', `${nome} (copia)`);
      if (!novoNome || !novoNome.trim()) return false;
      const res = await fetch(`/modelos/${encodeURIComponent(nome)}/duplicar`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ novo_nome: novoNome.trim() }),
      });
      const data = await res.json();
      if (!data.ok) { alert(data.erro || 'Falha ao duplicar.'); return false; }
      await carregarProjetos();
      return true;
    }

    async function excluirModelo(nome) {
      if (!confirm(`Excluir o modelo "${nome}"? Isso apaga TODAS as imagens, anotacoes e treinamentos dele. Nao da pra desfazer.`)) return false;
      const res = await fetch(`/modelos/${encodeURIComponent(nome)}/excluir`, { method: 'POST' });
      const data = await res.json();
      if (!data.ok) { alert(data.erro || 'Falha ao excluir.'); return false; }
      if (projetoAtual === nome) {
        projetoAtual = '';
        localStorage.removeItem('treinamento_projeto_atual');
      }
      await carregarProjetos();
      return true;
    }

    document.getElementById('btn-detalhe-renomear').addEventListener('click', async () => {
      if (await renomearModelo(projetoAtual)) {
        document.getElementById('detalhe-nome-projeto').textContent = projetoAtual;
        await carregarDetalheModelo();
      }
    });
    document.getElementById('btn-detalhe-duplicar').addEventListener('click', async () => {
      if (await duplicarModelo(projetoAtual)) alert('Modelo duplicado! Volte pra lista de Modelos pra abrir a copia.');
    });
    document.getElementById('btn-detalhe-excluir').addEventListener('click', async () => {
      if (await excluirModelo(projetoAtual)) mostrarListaModelos();
    });

    async function abrirModeloDetalhe(nome) {
      projetoAtual = nome;
      localStorage.setItem('treinamento_projeto_atual', projetoAtual);
      const selProjeto = document.getElementById('sel-projeto');
      if ([...selProjeto.options].some((o) => o.value === nome)) selProjeto.value = nome;
      await atualizarTudoDoProjeto();
      await carregarDetalheModelo();
      document.getElementById('modelos-lista-view').style.display = 'none';
      document.getElementById('modelos-detalhe-view').style.display = 'block';
    }

    let lightboxItens = [];
    let lightboxIndex = 0;

    async function carregarDetalheModelo() {
      document.getElementById('detalhe-nome-projeto').textContent = projetoAtual;
      const badge = document.getElementById('detalhe-badge-task');
      badge.textContent = projetoTask === 'segment' ? 'Segmentacao' : 'Deteccao';
      badge.className = `modelo-badge ${projetoTask}`;

      const pendentes = (ultimosDadosImagens.pendentes || []).length;
      const treino = (ultimosDadosImagens.treino || []).length;
      const validacao = (ultimosDadosImagens.validacao || []).length;
      const campos = [
        ['Tipo', projetoTask === 'segment' ? 'Segmentacao' : 'Deteccao'],
        ['Classes', state.classes.length],
        ['Pendentes', pendentes],
        ['Treino', treino],
        ['Validacao', validacao],
      ];
      document.getElementById('detalhe-dados-projeto').innerHTML = campos.map(([label, valor]) => `
        <div class="stat-cell"><span class="stat-label">${label}</span><span class="stat-value">${valor}</span></div>
      `).join('');

      const listaClassesEl = document.getElementById('detalhe-classes-lista');
      listaClassesEl.innerHTML = state.classes.length
        ? state.classes.map((nome, i) => `
            <div class="stat-row">
              <span class="stat-dot" style="background:${PALETTE[i % PALETTE.length]}"></span>
              <span class="stat-label">${nome}</span>
            </div>
          `).join('')
        : '<p class="hint" style="padding:8px 14px;">Nenhuma classe cadastrada.</p>';

      const res = await fetch(`/treinamento/resultados?${qp()}`);
      const resultados = await res.json();
      lightboxItens = [];
      resultados.forEach((r) => {
        r.imagens.forEach((img) => lightboxItens.push({ run: r.nome, imagem: img, dados: r }));
      });

      const wrap = document.getElementById('detalhe-resultados-lista');
      if (!resultados.length) {
        wrap.innerHTML = '<p class="hint">Nenhum treinamento concluido ainda pra esse modelo.</p>';
        return;
      }
      wrap.innerHTML = resultados.map((r) => {
        const m = r.metricas || {};
        const pills = [];
        if (m.map50 != null) pills.push(`mAP50: ${(m.map50 * 100).toFixed(1)}%`);
        if (m.precisao != null) pills.push(`Precisao: ${(m.precisao * 100).toFixed(1)}%`);
        if (m.recall != null) pills.push(`Recall: ${(m.recall * 100).toFixed(1)}%`);
        if (r.parametros && r.parametros.epochs != null) pills.push(`${r.parametros.epochs} epocas`);
        if (r.tempos && r.tempos.total) pills.push(`Tempo total: ${fmtDuracao(r.tempos.total)} (media ${fmtDuracao(r.tempos.media)}/epoca)`);
        return `
          <div class="run-card">
            <div class="run-card-header">
              <strong>${r.nome}</strong>
              <span class="hint">${r.modificado}</span>
            </div>
            ${pills.length ? `<div class="run-metric-pills">${pills.map((p) => `<span class="run-metric-pill">${p}</span>`).join('')}</div>` : ''}
            <div class="run-gallery">${r.imagens.map((img) => `<img data-run="${r.nome}" data-imagem="${img}" src="/treinamento/resultados/${r.nome}/${img}?${qp()}" loading="lazy" />`).join('')}</div>
            <div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:12px;">
              ${r.tem_modelo ? `
                <button class="btn-secondary" data-usar="${r.nome}">Usar este modelo</button>
                <a class="btn-secondary" style="text-decoration:none; display:inline-flex; align-items:center;" href="/treinamento/resultados/${r.nome}/modelo/baixar?${qp()}">Baixar modelo (.pt)</a>
              ` : '<span class="hint">Sem best.pt gerado</span>'}
            </div>
          </div>
        `;
      }).join('');

      wrap.querySelectorAll('.run-gallery img').forEach((img) => {
        img.addEventListener('click', () => {
          const run = img.getAttribute('data-run');
          const imagem = img.getAttribute('data-imagem');
          const idx = lightboxItens.findIndex((it) => it.run === run && it.imagem === imagem);
          abrirLightbox(idx);
        });
      });
      wrap.querySelectorAll('button[data-usar]').forEach((btn) => {
        btn.addEventListener('click', async () => {
          const run = btn.getAttribute('data-usar');
          const destinoPadrao = '../runs/detect/train/weights/best.pt';
          const destino = prompt('Copiar o modelo treinado para qual caminho?', destinoPadrao);
          if (destino === null) return;
          const r2 = await fetch(`/treinamento/aplicar/${run}`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ projeto: projetoAtual, destino }),
          });
          const d2 = await r2.json();
          alert(d2.ok ? `Modelo copiado para ${d2.destino}. Reinicie o programa que usa esse modelo.` : (d2.erro || 'Falha ao aplicar.'));
        });
      });
    }

    function abrirLightbox(idx) {
      if (idx < 0 || idx >= lightboxItens.length) return;
      lightboxIndex = idx;
      renderizarLightbox();
      document.getElementById('modal-lightbox').style.display = 'flex';
    }

    function renderizarLightbox() {
      const item = lightboxItens[lightboxIndex];
      if (!item) return;
      document.getElementById('lightbox-titulo').textContent = item.imagem;
      document.getElementById('lightbox-imagem').src = `/treinamento/resultados/${item.run}/${item.imagem}?${qp()}`;
      document.getElementById('lightbox-contador').textContent = `${lightboxIndex + 1} de ${lightboxItens.length}`;
      document.getElementById('lightbox-anterior').disabled = lightboxIndex <= 0;
      document.getElementById('lightbox-proxima').disabled = lightboxIndex >= lightboxItens.length - 1;

      const params = item.dados.parametros || {};
      const dadosTreino = [
        ['Execucao', item.run],
        ['Modificado', item.dados.modificado],
        ['Modelo base', params.model || '-'],
        ['Epocas', params.epochs ?? '-'],
        ['Batch', params.batch ?? '-'],
        ['Tamanho img.', params.imgsz ?? '-'],
      ];
      const tempos = item.dados.tempos || {};
      if (tempos.total) {
        dadosTreino.push(['Tempo total', fmtDuracao(tempos.total)], ['Media por epoca', fmtDuracao(tempos.media)],
                         ['Epoca mais rapida', fmtDuracao(tempos.mais_rapida)], ['Epoca mais lenta', fmtDuracao(tempos.mais_lenta)]);
      }
      document.getElementById('lightbox-dados-treino').innerHTML = dadosTreino.map(([label, valor]) => `
        <div class="stat-cell"><span class="stat-label">${label}</span><span class="stat-value">${valor}</span></div>
      `).join('');

      const m = item.dados.metricas || {};
      const metricasGrupo = document.getElementById('lightbox-metricas-grupo');
      const entradasMetricas = [
        ['Precisao', m.precisao], ['Recall', m.recall], ['mAP50', m.map50], ['mAP50-95', m.map50_95],
      ].filter(([, v]) => v != null);
      if (entradasMetricas.length) {
        metricasGrupo.style.display = 'block';
        document.getElementById('lightbox-metricas').innerHTML = entradasMetricas.map(([label, v]) => `
          <div class="stat-cell"><span class="stat-label">${label}</span><span class="stat-value">${(v * 100).toFixed(1)}%</span></div>
        `).join('');
      } else {
        metricasGrupo.style.display = 'none';
      }

      // tempo de cada epoca deste treinamento (lista compacta abaixo das metricas)
      let temposGrupo = document.getElementById('lightbox-tempos-grupo');
      if (!temposGrupo) {
        temposGrupo = document.createElement('div');
        temposGrupo.id = 'lightbox-tempos-grupo';
        temposGrupo.className = 'stat-group';
        document.getElementById('lightbox-download-wrap').before(temposGrupo);
      }
      temposGrupo.style.display = tempos.total ? 'block' : 'none';
      temposGrupo.innerHTML = tempos.total
        ? `<div class="stat-group-header">Tempo por epoca</div><div class="tempos-lista" style="margin-top:4px;">${
            tempos.epocas.map((s, i) => `<span class="${s > tempos.media * 1.5 ? 'lenta' : ''}">Epoca ${i + 1}<b>${fmtDuracao(s)}</b></span>`).join('')}</div>`
        : '';

      document.getElementById('lightbox-download-wrap').innerHTML = item.dados.tem_modelo
        ? `<a class="btn-secondary" style="text-decoration:none; display:block; text-align:center;" href="/treinamento/resultados/${item.run}/modelo/baixar?${qp()}">Baixar modelo (.pt)</a>`
        : '';
    }

    document.getElementById('lightbox-anterior').addEventListener('click', () => abrirLightbox(lightboxIndex - 1));
    document.getElementById('lightbox-proxima').addEventListener('click', () => abrirLightbox(lightboxIndex + 1));
    document.getElementById('modal-lightbox-fechar').addEventListener('click', () => {
      document.getElementById('modal-lightbox').style.display = 'none';
    });
    window.addEventListener('keydown', (ev) => {
      if (document.getElementById('modal-lightbox').style.display !== 'flex') return;
      if (ev.key === 'ArrowLeft') abrirLightbox(lightboxIndex - 1);
      if (ev.key === 'ArrowRight') abrirLightbox(lightboxIndex + 1);
      if (ev.key === 'Escape') document.getElementById('modal-lightbox').style.display = 'none';
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

    @app.get('/modelos')
    def listar_modelos():
        """Resumo de todos os projetos pra pagina 'Modelos' (lista geral
        com contagens, sem precisar abrir cada um pra saber o que tem
        dentro)."""
        resumos = [project_summary(get_project(nome)) for nome in list_projetos()]
        resumos.sort(key=lambda r: r['_ultima_modificacao_ts'], reverse=True)
        for r in resumos:
            del r['_ultima_modificacao_ts']
        return jsonify(resumos)

    @app.post('/modelos/<nome>/renomear')
    def renomear_modelo(nome):
        data = request.get_json(silent=True) or {}
        pp = get_project(nome)
        try:
            novo = rename_project(pp, str(data.get('novo_nome', '')))
        except ValueError as exc:
            return jsonify({'ok': False, 'erro': str(exc)}), 400
        return jsonify({'ok': True, 'projeto': novo.nome})

    @app.post('/modelos/<nome>/duplicar')
    def duplicar_modelo(nome):
        data = request.get_json(silent=True) or {}
        pp = get_project(nome)
        try:
            novo = duplicate_project(pp, str(data.get('novo_nome', '')))
        except ValueError as exc:
            return jsonify({'ok': False, 'erro': str(exc)}), 400
        return jsonify({'ok': True, 'projeto': novo.nome})

    @app.post('/modelos/<nome>/excluir')
    def excluir_modelo(nome):
        pp = get_project(nome)
        try:
            delete_project(pp)
        except ValueError as exc:
            return jsonify({'ok': False, 'erro': str(exc)}), 400
        return jsonify({'ok': True})

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

    @app.get('/cameras/<cam_id>')
    def obter_camera(cam_id):
        """Dados de uma camera pra preencher o formulario de edicao -- sem
        a senha (mesma logica de privacidade do /cameras GET geral: a
        senha nunca volta pro navegador ja preenchida). O campo de senha no
        formulario de edicao fica em branco e so e alterado se a pessoa
        digitar uma nova."""
        cam = camera_manager.cameras.get(cam_id)
        if not cam:
            return jsonify({'ok': False, 'erro': 'Camera nao encontrada.'}), 404
        return jsonify({
            'ok': True, 'id': cam_id, 'nome': cam['nome'], 'ip': cam['ip'],
            'usuario': cam['usuario'], 'porta': cam['porta'], 'caminho': cam['caminho'],
        })

    @app.post('/cameras/<cam_id>/editar')
    def editar_camera(cam_id):
        data = request.get_json(silent=True) or {}
        try:
            camera_manager.editar(
                cam_id, data.get('nome', ''), data.get('ip', ''), data.get('usuario', ''),
                data.get('senha', ''), data.get('porta', ''), data.get('caminho', ''),
            )
        except (ValueError, RuntimeError) as exc:
            return jsonify({'ok': False, 'erro': str(exc)}), 400
        return jsonify({'ok': True})

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

    @app.post('/imagens/importar')
    def importar_zip():
        """Importa um .zip no padrao YOLO (ex.: exportado pelo BeefVision):
        images/<nome>, labels/<nome>.txt e classes.txt. As marcacoes vem junto,
        ja convertidas para o tipo deste projeto (caixa ou poligono), e as
        classes sao casadas pelo nome (as que nao existirem sao criadas).
        destino=pendentes: a marcacao fica como 'pre-anotacao' para revisar;
        destino=treino/validacao: entra direto no dataset."""
        pp = get_project(request.form.get('projeto', ''))
        destino = request.form.get('destino', 'pendentes')
        destinos = {
            'pendentes': (pp.capturas, pp.capturas_labels),
            'treino': (pp.images_train, pp.labels_train),
            'validacao': (pp.images_val, pp.labels_val),
        }
        if destino not in destinos:
            return jsonify({'ok': False, 'erro': 'Destino invalido.'}), 400
        arquivo = request.files.get('arquivo')
        if not arquivo or not arquivo.filename:
            return jsonify({'ok': False, 'erro': 'Nenhum arquivo enviado.'}), 400
        try:
            z = zipfile.ZipFile(arquivo.stream)
        except zipfile.BadZipFile:
            return jsonify({'ok': False, 'erro': 'O arquivo nao e um .zip valido.'}), 400

        with z:
            membros = {}
            for info in z.infolist():
                if info.is_dir():
                    continue
                partes = info.filename.replace('\\', '/').split('/')
                membros['/'.join(p.lower() for p in partes[-2:])] = info
            # classes do arquivo -> classes do projeto (pelo nome)
            classes_zip = []
            info_classes = next((i for k, i in membros.items() if k.split('/')[-1] == 'classes.txt'), None)
            if info_classes:
                classes_zip = [c.strip() for c in z.read(info_classes).decode('utf-8-sig').splitlines() if c.strip()]
            classes = load_classes(pp)
            por_nome = {_normalizar_classe(c): i for i, c in enumerate(classes)}
            adicionadas = []
            mapa = {}
            for i, nome_cls in enumerate(classes_zip):
                chave = _normalizar_classe(nome_cls)
                if chave not in por_nome:
                    classes.append(nome_cls)
                    por_nome[chave] = len(classes) - 1
                    adicionadas.append(nome_cls)
                mapa[i] = por_nome[chave]
            if not classes_zip:  # sem classes.txt: usa os indices como estao
                mapa = {i: i for i in range(len(classes))}
            if adicionadas:
                save_classes(pp, classes)

            task = load_project_task(pp)
            img_dir, lbl_dir = destinos[destino]
            existentes = {p.name.lower() for d in (pp.capturas, pp.images_train, pp.images_val) for p in d.iterdir() if p.is_file()}
            importadas = ja_existiam = sem_marcacao = 0
            for chave, info in membros.items():
                pasta, nome = chave.split('/')[-2:] if '/' in chave else ('', chave)
                if pasta != 'images' or Path(nome).suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                    continue
                nome_real = safe_name(info.filename.replace('\\', '/').split('/')[-1])
                if nome_real.lower() in existentes:
                    ja_existiam += 1
                    continue
                linhas = []
                info_lbl = membros.get(f'labels/{Path(nome).stem}.txt')
                if info_lbl:
                    for linha in z.read(info_lbl).decode('utf-8-sig').splitlines():
                        convertida = converter_linha_yolo(linha, task, mapa)
                        if convertida:
                            linhas.append(convertida)
                else:
                    sem_marcacao += 1
                (img_dir / nome_real).write_bytes(z.read(info))
                if info_lbl or destino != 'pendentes':
                    (lbl_dir / (Path(nome_real).stem + '.txt')).write_text(
                        '\n'.join(linhas) + ('\n' if linhas else ''), encoding='utf-8')
                existentes.add(nome_real.lower())
                importadas += 1

        if not importadas and not ja_existiam:
            return jsonify({'ok': False, 'erro': 'Nenhuma imagem encontrada no .zip (esperado: pasta images/ e labels/).'}), 400
        return jsonify({'ok': True, 'importadas': importadas, 'ja_existiam': ja_existiam,
                        'sem_marcacao': sem_marcacao, 'classes_adicionadas': adicionadas})

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

    def _listar(pasta_dir: Path, label_dir=None, pre_anotacao=False):
        itens = []
        arquivos = []
        for ext in IMAGE_EXTENSIONS:
            arquivos.extend(pasta_dir.glob(ext))
        for p in sorted(arquivos, key=lambda f: f.stat().st_mtime, reverse=True):
            info = {'nome': p.name}
            if label_dir is not None:
                lbl = label_dir / (p.stem + '.txt')
                if pre_anotacao:
                    # pendentes: so marca as que vieram com marcacao previa (ex.: da IA)
                    if lbl.exists():
                        info['pre_anotado'] = True
                        info['classes'] = _classes_no_label(lbl)
                else:
                    info['anotado'] = lbl.exists() and lbl.stat().st_size > 0
                    info['classes'] = _classes_no_label(lbl)
            itens.append(info)
        return itens

    @app.get('/imagens')
    def listar_imagens():
        pp = get_project(request.args.get('projeto', ''))
        return jsonify({
            'pendentes': _listar(pp.capturas, pp.capturas_labels, pre_anotacao=True),
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
        mapping = {'pendentes': pp.capturas_labels, 'treino': pp.labels_train, 'validacao': pp.labels_val}
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

    @app.post('/anotar/varinha_magica')
    def anotar_varinha_magica():
        """Modo 'varinha magica': recebe um ponto clicado (coordenadas 0-1,
        relativas a imagem) e devolve o contorno detectado automaticamente
        ao redor do objeto naquele ponto, pra virar um poligono de
        segmentacao sem precisar clicar vertice por vertice."""
        data = request.get_json(silent=True) or {}
        pp = get_project(data.get('projeto', ''))
        path = _resolve_image_path(pp, data.get('pasta', ''), safe_name(data.get('nome', '')))
        if path is None or not path.exists() or not path.is_file():
            return jsonify({'ok': False, 'erro': 'Imagem nao encontrada.'}), 404

        try:
            x_norm = float(data.get('x'))
            y_norm = float(data.get('y'))
        except (TypeError, ValueError):
            return jsonify({'ok': False, 'erro': 'Coordenadas invalidas.'}), 400
        if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
            return jsonify({'ok': False, 'erro': 'Coordenadas fora da imagem.'}), 400

        # raio_percentual: tamanho do recorte (em % da menor dimensao da
        # imagem) onde o GrabCut roda, centrado no clique -- ajuste pra
        # ficar um pouco maior que o objeto que voce quer selecionar (ver
        # docstring de varinha_magica_contorno pra mais detalhes).
        try:
            raio_percentual = float(data.get('raio_percentual', 20))
        except (TypeError, ValueError):
            raio_percentual = 20.0
        raio_percentual = min(max(raio_percentual, 3.0), 60.0)

        img = cv2.imread(str(path))
        if img is None:
            return jsonify({'ok': False, 'erro': 'Falha ao ler a imagem.'}), 500
        h, w = img.shape[:2]
        seed_x = min(max(int(round(x_norm * w)), 0), w - 1)
        seed_y = min(max(int(round(y_norm * h)), 0), h - 1)
        raio_px = max(20, int(min(h, w) * (raio_percentual / 100.0)))

        pontos = varinha_magica_contorno(img, seed_x, seed_y, raio_px)
        if pontos is None:
            return jsonify({
                'ok': False,
                'erro': 'Nao foi possivel isolar um contorno nesse ponto. Tente clicar mais no centro do objeto, ou ajustar a area de busca.',
            }), 400

        pontos_norm = [[px / w, py / h] for px, py in pontos]
        return jsonify({'ok': True, 'pontos': pontos_norm})

    @app.post('/anotar')
    def anotar():
        data = request.get_json(silent=True) or {}
        pp = get_project(data.get('projeto', ''))
        pasta_origem = data.get('pasta', 'pendentes')
        nome = safe_name(data.get('nome', ''))
        destino = data.get('destino', 'treino')
        caixas = data.get('caixas', [])

        origem_map = {
            'pendentes': (pp.capturas, pp.capturas_labels),  # a pre-anotacao sai daqui ao salvar
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
        mapping = {
            'pendentes': (pp.capturas, pp.capturas_labels),
            'treino': (pp.images_train, pp.labels_train),
            'validacao': (pp.images_val, pp.labels_val),
        }
        destinos_validos = {'treino', 'validacao'}
        if origem not in mapping or destino not in destinos_validos or origem == destino or not nome:
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
            'pendentes': (pp.capturas, pp.capturas_labels),
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

    @app.get('/classes/uso')
    def uso_classes():
        """Quantas marcacoes e imagens cada classe tem (mostrado nos botoes de classe e no aviso de remover)."""
        pp = get_project(request.args.get('projeto', ''))
        uso = contar_uso_classes(pp)
        return jsonify([dict(nome=n, **uso.get(i, {'marcacoes': 0, 'imagens': 0}))
                        for i, n in enumerate(load_classes(pp))])

    def _classe_pedida(pp, data):
        """Confere indice + nome (evita mexer na classe errada se a tela estiver desatualizada)."""
        classes = load_classes(pp)
        try:
            indice = int(data.get('indice'))
        except (TypeError, ValueError):
            return None, classes, 'Classe invalida.'
        if not (0 <= indice < len(classes)) or classes[indice] != str(data.get('nome', '')):
            return None, classes, 'A lista de classes mudou. Atualize a pagina e tente de novo.'
        return indice, classes, None

    @app.post('/classes/renomear')
    def renomear_classe():
        data = request.get_json(silent=True) or {}
        pp = get_project(data.get('projeto', ''))
        indice, classes, erro = _classe_pedida(pp, data)
        if erro:
            return jsonify({'ok': False, 'erro': erro}), 400
        novo = str(data.get('novo_nome', '')).strip()
        if not novo:
            return jsonify({'ok': False, 'erro': 'Nome vazio.'}), 400
        if novo in classes and classes.index(novo) != indice:
            return jsonify({'ok': False, 'erro': 'Ja existe uma classe com esse nome.'}), 400
        # so o nome muda: as marcacoes guardam o numero da classe, que continua o mesmo
        classes[indice] = novo
        save_classes(pp, classes)
        return jsonify({'ok': True, 'classes': classes})

    @app.post('/classes/remover')
    def remover_classe():
        data = request.get_json(silent=True) or {}
        pp = get_project(data.get('projeto', ''))
        if training_state.running and training_state.projeto == pp.nome:
            return jsonify({'ok': False, 'erro': 'Ha um treinamento rodando neste modelo. Aguarde terminar para remover a classe.'}), 409
        indice, classes, erro = _classe_pedida(pp, data)
        if erro:
            return jsonify({'ok': False, 'erro': erro}), 400
        try:
            resultado = remover_classe_do_dataset(pp, indice)
        except OSError as exc:
            return jsonify({'ok': False, 'erro': f'Falha ao alterar as marcacoes: {exc}'}), 500
        return jsonify({'ok': True, **resultado, 'classes': load_classes(pp)})

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
                    'metricas': read_run_metrics(d),
                    'tempos': read_run_tempos(d),
                    'parametros': read_run_args(d),
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
        # Se ja tinha um teste rodando (ex.: a pessoa fechou a aba sem
        # clicar em "Parar"), para ele sozinho em vez de recusar iniciar um
        # novo -- senao fica um teste "preso" que ninguem mais consegue
        # parar pela pagina (a unica forma seria reiniciar o programa).
        if test_video_state.running:
            print('[INFO] Novo teste de video solicitado com um teste anterior ainda em andamento -- parando o anterior automaticamente.')
            test_video_state.stop()

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
