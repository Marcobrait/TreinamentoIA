# Treinamento de Modelo - Contador IA

Ferramenta web independente para capturar imagens da câmera, anotar as
caixas (bounding boxes) direto no navegador, gerenciar o dataset de
treino/validação e treinar novos modelos YOLO — sem depender de sites
externos (ex: makesense.ai). Suporta **vários modelos separados** (ex:
um para caixas abertas/fechadas, outro para produtos de mercado, etc.),
cada um com suas próprias imagens, classes e histórico de treinamento.

## Fluxo de trabalho

1. **Escolher/criar um modelo**: no topo da página tem um seletor com os
   modelos já existentes. Clique em "+ Novo modelo" para criar outro do
   zero — dê um nome (ex: `Treinamento Mercados`) e, se quiser, já liste
   as classes iniciais separadas por vírgula (ex: `PULL, ESTANTE`). Pode
   deixar sem nenhuma classe e cadastrar depois, ou só com uma. O
   programa cria a pasta `projetos/<nome do modelo>/` pra guardar tudo
   daquele modelo separado dos outros.
2. **Capturar**: com a câmera ao vivo na tela, clique em "Capturar frame"
   quantas vezes quiser. Cada clique salva um frame nas imagens
   pendentes **do modelo selecionado no momento**.
3. **Cadastrar classes**: dê nome às classes que esse modelo vai aprender
   a reconhecer. A ordem de cadastro define o índice da classe (a
   primeira é `0`, a segunda é `1`, etc.) — por isso não é possível
   excluir uma classe pelo momento, só adicionar.
4. **Anotar**: na aba "Pendentes", clique em "Anotar" numa imagem. Uma
   janela abre com a imagem — selecione a classe ativa (à direita) e
   desenhe as caixas na imagem clicando e arrastando o mouse. Escolha se
   a imagem vai para "Treino" ou "Validação" e clique em "Salvar
   anotação". Isso grava o `.txt` no formato YOLO e move a imagem para a
   pasta correspondente.
5. **Gerenciar**: nas abas "Treino"/"Validação" dá pra reabrir uma imagem
   para reanotar, mover ela para o outro grupo, ou excluir.
6. **Treinar**: configure épocas, tamanho de imagem, batch, workers e o
   modelo base, e clique em "Iniciar treinamento". A página mostra logo
   no topo do formulário se foi detectada GPU (e qual) ou se vai rodar
   na CPU, e acompanha o progresso em tempo real (época atual, métricas,
   log, e o dispositivo usado). Só roda um treinamento por vez no
   servidor, não importa de qual modelo.
7. **Resultados**: ao concluir, os gráficos/resultados aparecem na seção
   "Resultados". O botão "Usar este modelo" pergunta o caminho de
   destino (já vem preenchido com
   `../runs/detect/train/weights/best.pt`, o que o `web_contador3.py`
   espera) e copia o `best.pt` treinado pra lá. É só reiniciar o
   programa que consome aquele modelo depois.

## Estrutura de pastas

```
web_treinamento.py       programa principal
config.cfg                 configuracao real (camera + porta web) - NAO versionado
config.cfg.example         modelo de configuracao
yolov8n.pt                 checkpoint base (para treinar offline, sem precisar baixar)
projetos/
  BTS/                          um modelo = uma pasta (nome dado ao criar)
    dataset/
      capturas/                  imagens capturadas, ainda sem anotacao
      classes.txt                lista de classes deste modelo (indice = ordem)
      images/train, images/val    imagens ja anotadas, por conjunto
      labels/train, labels/val    os .txt YOLO correspondentes
      data.yaml                   gerado automaticamente a cada treino
    runs_treinamento/            saida de cada treinamento deste modelo (pesos, graficos)
  Treinamento Mercados/          outro modelo, com a mesma estrutura, isolado do BTS
    dataset/...
    runs_treinamento/...
```

Se você já vinha usando uma versão anterior desta ferramenta (com
`dataset/` e `runs_treinamento/` direto na raiz, sem suporte a vários
modelos), na primeira vez que rodar essa versão nova ele **migra
automaticamente** o que já existia para `projetos/BTS/`, sem perder
nada.

## Instalação

Mesma lógica do pacote principal:

1. Copie `config.cfg.example` para `config.cfg` e preencha os dados da
   câmera (pode ser a mesma câmera do `web_contador3.py`).
2. Rode `instalar.bat` (uma vez).
3. Rode `iniciar.bat`.
4. Acesse `http://localhost:5011` (porta configurável em `[web] porta`).

Pode rodar ao mesmo tempo que o `web_contador3.py` — só usam portas
diferentes (5011 aqui, 5003 lá por padrão) e cada um abre sua própria
conexão RTSP com a câmera.

## GPU NVIDIA

O `instalar.bat` detecta sozinho, na hora da instalação, se o servidor
tem uma placa de vídeo NVIDIA (checando se o comando `nvidia-smi` existe
— ele vem junto do driver da NVIDIA). Se detectar, troca automaticamente
o `torch`/`torchvision` que tinham sido instalados para CPU pela versão
com suporte a CUDA, e o treinamento passa a usar a placa de vídeo em vez
do processador — não precisa fazer mais nada. Se não detectar nenhuma
GPU, mantém a versão para CPU normalmente. A página de treinamento
sempre mostra qual dos dois está em uso (GPU detectada ou CPU) tanto
antes de iniciar quanto durante o treinamento.

Detalhes e ajustes:

- Por padrão o `instalar.bat` instala o build `cu126` (CUDA 12.6), que
  funciona com a maioria das placas/drivers NVIDIA recentes (driver
  NVIDIA relativamente atualizado). Se sua placa/driver precisar de
  outra versão de CUDA, ou se quiser conferir qual é a recomendada pro
  seu caso, veja https://pytorch.org/get-started/locally/ e rode
  manualmente o comando de instalação equivalente (trocando `cu126`
  pela versão certa) dentro do ambiente virtual:
  ```powershell
  .venv\Scripts\python.exe -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
  ```
- Se a troca para a versão com CUDA falhar (ex: erro de certificado SSL
  mesmo depois da tentativa automática com `--trusted-host`, ou falta de
  internet), o `instalar.bat` avisa na tela e **segue funcionando
  normalmente com CPU** — não trava a instalação.
- Se aparecer um erro do tipo `Could not find a version that satisfies
  the requirement torch==X.Y.Z` durante essa etapa, é porque o índice
  `cuXXX` fixado no `instalar.bat` parou de publicar builds para a
  versão do `torch`/`torchvision` que está no `requirements.txt` (o
  PyTorch descontinua índices CUDA antigos aos poucos). Para corrigir,
  veja em `https://download.pytorch.org/whl/cuXXX/torch/` (trocando
  `cuXXX` por `cu124`, `cu126`, `cu128` etc.) qual índice ainda lista a
  versão exata usada no projeto, e atualize as duas linhas
  `--index-url https://download.pytorch.org/whl/cuXXX` dentro do
  `instalar.bat` para esse índice.
- **Campo "Workers"** (na página de treinamento): controla quantos
  processos carregam/preparam imagens em paralelo durante o treino.
  Valores mais altos usam mais RAM (inclusive RAM da própria placa de
  vídeo, quando treinando por GPU) e podem deixar o treino mais rápido;
  porém, se a GPU tiver pouca memória, um valor alto de workers pode
  fazer o treinamento **travar ou fechar sozinho no meio** (erro comum
  em GPUs com poucos GB de VRAM). Se isso acontecer, reduza o valor de
  workers (experimente 2, depois 1, depois 0) e tente treinar de novo.

## Erro ao iniciar: falha ao carregar DLL do torch (WinError 1114)

Se o `iniciar.bat` (ou `instalar.bat`) mostrar um erro assim:
```
OSError: [WinError 1114] A dynamic link library (DLL) initialization
routine failed. Error loading "...\torch\lib\c10.dll" ...
```
Isso **não é** o mesmo problema de proxy/certificado — é o Windows desse
servidor não ter o **Microsoft Visual C++ Redistributable** instalado
(as DLLs do `torch` dependem dele). Instale:

👉 https://aka.ms/vs/17/release/vc_redist.x64.exe (link oficial da Microsoft)

Depois de instalar, reinicie o servidor (ou abra um prompt novo) e rode
`iniciar.bat` de novo.

Se o erro persistir mesmo depois de instalar o Redistributable, a causa
provável é o processador do servidor ser antigo demais e não suportar
AVX2 (exigido pelos builds recentes do PyTorch) — nesse caso é preciso
usar uma versão do `torch` compilada sem essa exigência.

## Instalação offline (rede corporativa sem internet confiável)

Em redes corporativas com proxy/firewall que inspeciona HTTPS, o
`instalar.bat` já tenta contornar erros de certificado sozinho (veja o
aviso que ele mostra na tela). Mas em alguns casos esse mesmo proxy
corrompe o download de arquivos grandes (ex: `torch`, `polars-runtime-32`,
`opencv`) mesmo com o certificado contornado — o pip não avisa erro, mas
o pacote fica instalado pela metade e quebra na hora de usar (ex: erro
"Polars binary is missing!" ou algo travando no meio do treinamento).

Se isso acontecer, a solução é baixar os pacotes numa rede boa (sua
máquina, por exemplo) e levar os arquivos prontos pro servidor, sem o
servidor precisar baixar nada da internet.

**1. Na sua máquina** (com internet normal, fora da rede corporativa),
dentro desta mesma pasta (`deploy\treinamento`):
```powershell
.venv\Scripts\python.exe -m pip download -r requirements.txt -d pacotes_offline
```
Isso cria uma pasta `pacotes_offline\` com todos os `.whl` já baixados
por completo (cerca de 40 arquivos, alguns grandes — o `torch` sozinho
tem mais de 100 MB). Confira quantos arquivos vieram:
```powershell
Get-ChildItem pacotes_offline | Measure-Object | Select-Object -ExpandProperty Count
```

**2. Copie a pasta `pacotes_offline`** inteira para o servidor (pendrive,
compartilhamento de rede, etc.), dentro da mesma pasta `deploy\treinamento`.

**3. No servidor**, instale a partir dessa pasta, sem tocar na internet:
```powershell
.venv\Scripts\python.exe -m pip install --no-index --find-links=pacotes_offline -r requirements.txt
```
`--no-index` impede o pip de tentar falar com a internet; `--find-links`
manda ele usar só os arquivos da pasta local.

**Se já tinha instalado antes e algum pacote ficou corrompido**
(erro tipo `Cannot uninstall ... no RECORD file was found`), a forma mais
segura é recriar o ambiente virtual do zero em vez de tentar consertar
pacote por pacote:
```powershell
Remove-Item -Recurse -Force .venv
python -m venv .venv
.venv\Scripts\python.exe -m pip install --no-index --find-links=pacotes_offline -r requirements.txt
```
(`Remove-Item` aí só apaga o ambiente virtual — pacotes instalados. Não
mexe no `config.cfg`, nos `projetos\` nem em nada que você criou.)

## Sobre os modelos base

- **YOLOv8n**: já vem incluso (`yolov8n.pt`), funciona sem internet.
- **YOLOv8s / YOLOv8m**: mais precisos, porém maiores — a biblioteca
  tenta baixá-los automaticamente na primeira vez que forem usados, o
  que exige acesso à internet (se o servidor estiver numa rede
  corporativa com proxy/SSL, isso pode falhar — veja o
  `LEIA-ME.txt`/`README.md` do pacote principal para o mesmo problema
  com o `pip`).
- **Continuar do último treinamento**: reaproveita o `last.pt` do
  treinamento mais recente **desse mesmo modelo** como ponto de partida,
  em vez de começar do zero — útil para refinar um modelo já treinado
  com imagens novas.

## Limitações conhecidas

- Não dá para excluir um modelo (pasta em `projetos/`) nem uma classe
  pela interface — se precisar remover algum, apague a pasta/edite o
  `classes.txt` manualmente com o programa fechado.
- Não dá para cancelar um treinamento em andamento pela interface (a
  biblioteca de treinamento não oferece um jeito limpo de interromper no
  meio) — se precisar parar, feche a janela do `iniciar.bat`.
- Só um treinamento por vez no servidor inteiro (não importa de qual
  modelo) — iniciar um novo enquanto outro roda é bloqueado.
- Treinamento sem GPU (CPU only) é bem mais lento — poucas épocas com
  imagens pequenas já podem levar bastante tempo. Ajuste `epocas` e
  `imgsz` conforme o hardware disponível. O `instalar.bat` já detecta e
  usa GPU NVIDIA automaticamente quando disponível (veja a seção
  "GPU NVIDIA" acima).
