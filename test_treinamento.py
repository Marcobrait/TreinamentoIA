"""Verifica importação e despacho sem baixar pesos ou executar treino real."""
import io
import json
import tempfile
import unittest
import zipfile
import re
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import treinamento as app


def zip_dataset(classes=range(1, 10), fonte='recortes', split='val'):
    stream = io.BytesIO()
    ok, img = cv2.imencode('.png', np.zeros((32, 32, 3), dtype=np.uint8))
    assert ok
    with zipfile.ZipFile(stream, 'w') as z:
        z.writestr('manifesto.json', json.dumps({'formato': 'marmoreio-classify-v1', 'fonte': fonte}))
        for classe in classes:
            for s in ('train', split):
                z.writestr(f'{s}/{classe}/foto-{classe}-{s}.png', img.tobytes())
    stream.seek(0)
    return stream


class TreinamentoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old_root = app.PROJETOS_DIR
        app.PROJETOS_DIR = Path(self.temp.name)
        self.client = app.create_app(None, 'Teste').test_client()

    def tearDown(self):
        app.PROJETOS_DIR = self.old_root
        self.temp.cleanup()

    def test_importa_nove_classes_e_nao_duplica(self):
        pp = app.create_project('Teste', task='classify')
        resultado = app.importar_classificacao(pp, zip_dataset())
        self.assertEqual(resultado['importadas'], 18)
        self.assertEqual(app.load_classes(pp), [str(i) for i in range(1, 10)])
        app.validar_dataset_classificacao(pp)
        self.assertEqual(app.importar_classificacao(pp, zip_dataset())['repetidas'], 18)
        self.assertEqual(app._resolve_base_model(pp, 'yolov8n-cls.pt'), 'yolov8n-cls.pt')
        self.assertEqual(app.project_summary(pp)['total_treino'], 9)

    def test_importa_valid_e_recusa_mistura_de_fontes(self):
        pp = app.create_project('Teste', task='classify')
        app.importar_classificacao(pp, zip_dataset(split='valid'))
        app.validar_dataset_classificacao(pp)
        with self.assertRaisesRegex(ValueError, 'outra fonte'):
            app.importar_classificacao(pp, zip_dataset(fonte='originais'))

    def test_permite_iniciar_com_classes_sem_imagens(self):
        pp = app.create_project('Teste', task='classify')
        app.importar_classificacao(pp, zip_dataset(classes=[1]))
        with patch.object(app.threading, 'Thread') as thread:
            resposta = self.client.post('/treinamento/iniciar', json={
                'projeto': pp.nome, 'modelo_base': 'yolov8n-cls.pt'})
            self.assertEqual(resposta.status_code, 200, resposta.json)
            thread.return_value.start.assert_called_once()
        app.training_state.finish(pp.runs / 'fake')

    def test_classes_esparsas_preservam_notas_e_pastas_vazias_nao_bloqueiam(self):
        pp = app.create_project('Teste', task='classify')
        app.importar_classificacao(pp, zip_dataset(classes=[1, 5, 9]))
        # A classe 9 fica sem validação; o preparo deve permitir e registrar o reaproveitamento.
        (pp.classify_dataset / 'val' / '9' / 'foto-9-val.png').unlink()
        destino = app.preparar_dataset_classificacao(pp, pp.runs / 'teste' / 'dataset')
        self.assertEqual(sorted(d.name for d in (destino / 'train').iterdir()), ['1','5','9'])
        self.assertEqual(sorted(d.name for d in (destino / 'val').iterdir()), ['1','5','9'])
        from ultralytics.data.utils import check_cls_dataset
        from torchvision.datasets import ImageFolder
        self.assertEqual(check_cls_dataset(str(destino))['names'], {0:'1',1:'5',2:'9'})
        self.assertEqual(ImageFolder(str(destino / 'train')).class_to_idx, {'1':0,'5':1,'9':2})
        self.assertEqual(ImageFolder(str(destino / 'val')).class_to_idx, {'1':0,'5':1,'9':2})
        registro = json.loads((destino.parent / 'classes_treinadas.json').read_text(encoding='utf-8'))
        self.assertEqual(registro['validacao_reutiliza_treino'], ['9'])
        self.assertIn('2', registro['classes_sem_treino'])
        self.assertEqual(app.load_classes(pp), [str(i) for i in range(1,10)])

    def test_dataset_sem_nenhuma_imagem_de_treino_continua_explicito(self):
        pp = app.create_project('Teste', task='classify')
        resposta = self.client.post('/treinamento/iniciar', json={'projeto':pp.nome,'modelo_base':'yolov8n-cls.pt'})
        self.assertEqual(resposta.status_code, 400)
        self.assertIn('pelo menos uma imagem', resposta.json['erro'])

    def test_rotas_importacao_resumo_e_download(self):
        pp = app.create_project('Teste', task='classify')
        resposta = self.client.post('/classificacao/importar', data={
            'projeto': pp.nome, 'arquivo': (zip_dataset(), 'dataset.zip')})
        self.assertEqual(resposta.status_code, 200, resposta.json)
        self.assertEqual(self.client.get('/classificacao/resumo?projeto=Teste').json['contagens']['train']['9'], 1)
        pesos = pp.runs / 'teste' / 'weights'
        pesos.mkdir(parents=True)
        (pesos / 'best.pt').write_bytes(b'fake')
        resposta = self.client.get('/treinamento/resultados/teste/modelo/baixar?projeto=Teste')
        self.assertIn('marmoreio.pt', resposta.headers['Content-Disposition'])
        resposta.close()

    def test_detect_segment_preservados_e_classify_despacha_diretorio(self):
        for task, modelo in [('detect', 'yolov8n.pt'), ('segment', 'yolov8n-seg.pt'), ('classify', 'yolov8n-cls.pt')]:
            pp = app.create_project(task, classes_iniciais=['1', '2'], task=task)
            if task == 'classify':
                app.importar_classificacao(pp, zip_dataset())
            else:
                for pasta in (pp.images_train, pp.images_val):
                    (pasta / 'foto.jpg').write_bytes(b'fake')
            chamadas = []
            class Modelo:
                def __init__(self, caminho): chamadas.append(caminho)
                def add_callback(self, *args): pass
                def train(self, **kwargs): chamadas.append(kwargs)
            params = dict(epocas=1, imgsz=224, batch=1, workers=0, modelo_base=modelo)
            app.training_state.reset_for_start(pp.nome, 1)
            with patch.object(app, 'YOLO', Modelo):
                app._run_training(pp, params)
            self.assertEqual(len(chamadas), 2, app.training_state.snapshot())
            data = Path(chamadas[1]['data'])
            self.assertEqual(data.is_dir(), task == 'classify')
            if task != 'classify':
                self.assertEqual(app.yaml.safe_load(data.read_text())['names'], {0: '1', 1: '2'})

    def test_rejeita_caminho_malicioso(self):
        pp = app.create_project('Teste', task='classify')
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as z:
            z.writestr('../train/1/foto.jpg', b'fake')
        stream.seek(0)
        with self.assertRaisesRegex(ValueError, 'Caminho'):
            app.importar_classificacao(pp, stream)

    def test_javascript_da_interface_tem_sintaxe_valida(self):
        scripts = re.findall(r'<script>(.*?)</script>', app.PAGE_HTML, re.S)
        self.assertTrue(scripts)
        for script in scripts:
            resultado = subprocess.run(['node', '--check'], input=script, text=True, capture_output=True)
            self.assertEqual(resultado.returncode, 0, resultado.stderr)

    def test_galeria_paginada_e_imagem_com_nome_repetido_em_classes(self):
        pp = app.create_project('Teste', task='classify')
        app.importar_classificacao(pp, zip_dataset())
        for i in range(45):
            (pp.classify_dataset / 'train' / '1' / f'extra-{i}.png').write_bytes(b'imagem')
        resposta = self.client.get('/classificacao/imagens?projeto=Teste&pasta=treino&por_pagina=10&pagina=2&classe=1')
        self.assertEqual(resposta.status_code, 200)
        self.assertEqual(len(resposta.json['itens']), 10)
        self.assertEqual(resposta.json['paginacao']['total'], 46)
        self.assertEqual(resposta.json['paginacao']['total_paginas'], 5)
        self.assertEqual(resposta.json['contagens']['validacao'], 9)
        foto = self.client.get('/classificacao/imagem?projeto=Teste&chave=train/1/foto-1-train.png')
        self.assertEqual(foto.status_code, 200)
        foto.close()
        self.assertEqual(self.client.get('/classificacao/imagem?projeto=Teste&chave=../config.cfg').status_code, 404)

    def test_move_e_exclui_selecao_sem_sobrescrever_imagem_destino(self):
        pp = app.create_project('Teste', task='classify')
        app.importar_classificacao(pp, zip_dataset())
        origem = pp.classify_dataset / 'train' / '1' / 'foto-1-train.png'
        destino = pp.classify_dataset / 'val' / '1' / origem.name
        destino.write_bytes(b'preservar')
        mover = self.client.post('/classificacao/editar', json={'projeto':'Teste', 'acao':'mover',
            'destino':'validacao', 'chaves':['train/1/foto-1-train.png','train/2/foto-2-train.png']})
        self.assertEqual(mover.status_code, 200, mover.json)
        self.assertEqual(mover.json['alteradas'], 2)
        self.assertEqual(destino.read_bytes(), b'preservar')
        self.assertFalse(origem.exists())
        excluir = self.client.post('/classificacao/editar', json={'projeto':'Teste', 'acao':'excluir',
            'chaves':['val/2/foto-2-train.png']})
        self.assertEqual(excluir.status_code, 200)
        self.assertFalse((pp.classify_dataset / 'val' / '2' / 'foto-2-train.png').exists())
        self.assertTrue(list((pp.root / '.lixeira_classificacao').rglob('foto-2-train.png')))

    def test_upload_avulso_pendente_rotula_move_e_bloqueia_durante_treino(self):
        pp = app.create_project('Teste', task='classify')
        ok, png = cv2.imencode('.png', np.zeros((32,32,3), dtype=np.uint8))
        resposta = self.client.post('/classificacao/upload', data={'projeto':'Teste',
            'destino':'pendentes','arquivos':(io.BytesIO(png.tobytes()),'foto.png')})
        self.assertEqual(resposta.status_code, 200, resposta.json)
        listar = self.client.get('/classificacao/imagens?projeto=Teste&pasta=pendentes').json
        chave = listar['itens'][0]['chave']
        sem_classe = self.client.post('/classificacao/editar', json={'projeto':'Teste',
            'acao':'mover','destino':'treino','chaves':[chave]})
        self.assertEqual(sem_classe.status_code, 400)
        rotular = self.client.post('/classificacao/editar', json={'projeto':'Teste',
            'acao':'mover','destino':'treino','classe':'9','chaves':[chave]})
        self.assertEqual(rotular.status_code, 200, rotular.json)
        self.assertTrue((pp.classify_dataset / 'train' / '9' / 'foto.png').exists())
        app.training_state.reset_for_start('Teste',1)
        try:
            bloqueado = self.client.post('/classificacao/editar', json={'projeto':'Teste','acao':'excluir','chaves':['train/9/foto.png']})
            self.assertEqual(bloqueado.status_code, 409)
        finally:
            app.training_state.finish(pp.runs / 'fake')

    def test_upload_avulso_valida_lote_antes_de_gravar(self):
        pp = app.create_project('Teste', task='classify')
        ok, png = cv2.imencode('.png', np.zeros((32,32,3), dtype=np.uint8))
        resultado = self.client.post('/classificacao/upload', data={'projeto':'Teste','destino':'treino','classe':'3',
            'arquivos':[(io.BytesIO(png.tobytes()),'valida.png'),(io.BytesIO(b'invalida'),'outra.png')]})
        self.assertEqual(resultado.status_code, 400)
        self.assertFalse((pp.classify_dataset / 'train' / '3' / 'valida.png').exists())
        resultado = self.client.post('/classificacao/editar',json={'projeto':'Teste','acao':'mover',
            'destino':'treino','chaves':['../../arquivo.png']})
        self.assertEqual(resultado.status_code, 400)


if __name__ == '__main__' and '--preview' in sys.argv:
    class Cameras:
        active_id = None
        cameras = {}
        def listar(self): return []
        def status(self): return {}
        def get_frame(self): return None
    with tempfile.TemporaryDirectory(prefix='classify-preview-') as pasta:
        app.PROJETOS_DIR = Path(pasta)
        pp = app.create_project('MarmoreioClassify', task='classify')
        app.importar_classificacao(pp, zip_dataset())
        for i in range(45):
            img = np.full((180,240,3),(45+i*2,70+i,110+i),dtype=np.uint8)
            cv2.putText(img,str(i+1),(60,110),cv2.FONT_HERSHEY_SIMPLEX,2,(240,240,240),3)
            cv2.imwrite(str(pp.classify_dataset / 'train' / str(i%9+1) / f'amostra-{i}.png'),img)
        web = app.create_app(Cameras(),'Prévia com imagens sintéticas')
        @web.before_request
        def delay_upload():
            if app.request.path == '/classificacao/upload':
                time.sleep(3)
        web.run(host='127.0.0.1',port=5099,threaded=True)
elif __name__ == '__main__':
    unittest.main()
