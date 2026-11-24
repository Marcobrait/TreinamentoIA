# Classificação de marmoreio

Execute `iniciar-treinamento.bat` nesta pasta. Ele usa `treinamento.py`, baseado em
`web_treinamento2 - Copia.py`. Os iniciadores e treinadores existentes foram preservados.
O novo programa mantém os projetos existentes, detecção, segmentação, câmeras,
anotações, acompanhamento, resultados e testador; classificação usa um dataset separado.
Evite executar dois treinadores simultaneamente na mesma porta/projetos.

1. No Marmoreio, abra Imagens → Exportar dataset.
2. Para detectar e salvar as regiões antes de exportar, selecione “Detectar com IA e salvar as regiões do filtro”.
   Aguarde o progresso: o sistema salva recortes, máscaras e metadados e solicita o ZIP ao terminar.
   A detecção usa todas as fotos com classe 1 a 9 do filtro, inclusive as de outras páginas.
   Falhas ficam fora do treino e são listadas no manifesto do ZIP. Notas e originais são preservados.
   Para reutilizar recortes existentes sem repetir a inferência, selecione “Regiões de interesse já salvas”.
   Fotos sem recorte salvo são ignoradas e listadas no manifesto; o botão Detectar com IA
   faz apenas uma consulta. Use fotos originais somente quando esse for o dado de treino desejado.
3. Mantenha marcada a separação treino/validação. O ZIP contém `train/1` até `train/9`,
   `val/1` até `val/9`, `classes.txt`, `manifesto.json` e `dataset.csv`.
4. No novo treinador, clique em Novo modelo e selecione Classificação.
5. No painel Dataset de classificação, importe o ZIP e confira as quantidades.
   O treino usa somente as classes que têm imagens em Treino, mantendo seus nomes.
   Classes vazias não bloqueiam o início e ficam fora do modelo dessa execução.
   O importador aceita também
   um ZIP genérico `train/<classe>/foto` e `val/<classe>/foto` (ou `valid`).
   Durante o envio, o indicador mostra a porcentagem transferida; depois acompanha
   a verificação no servidor. A galeria e o resumo usam skeleton enquanto carregam.
   Em “Adicionar imagens avulsas”, envie JPG/PNG escolhendo classe e destino.
   Sem classe, envie para Pendentes e rotule na revisão.
6. Escolha YOLOv8n-cls (ou s/m) em Modelo base e inicie. O padrão de classificação
   é 224 pixels; épocas, batch e workers continuam ajustáveis no mesmo formulário.
   Na primeira utilização, pesos -cls ausentes precisam ser baixados pelo Ultralytics.
7. Confira resultados e teste uma imagem. Ao baixar o modelo de nove classes numéricas,
   o arquivo se chama `marmoreio.pt`. Os pesos ficam em `projetos/<nome>/runs_treinamento/<treino>/weights/best.pt`.
8. Copie para `classificador-service/modelos/marmoreio.pt` no sistema Marmoreio.
   Deixe `MAPA_CLASSES=` vazio para respeitar os nomes numéricos; se alterar o .env,
   reinicie o serviço. Use “Recarregar modelo” para carregar pesos novos.

| Nome da classe no modelo | Exibição no Marmoreio |
|---|---|
| 1 | M1 · Classe 1 |
| 2 | M1 · Classe 2 |
| 3 | M2 · Classe 3 |
| 4 | M2 · Classe 4 |
| 5 | M3 · Classe 5 |
| 6 | M3 · Classe 6 |
| 7 | M4 · Classe 7 |
| 8 | M5 · Classe 8 |
| 9 | M5 · Classe 9 |

O índice interno do YOLO começa em zero, mas o sistema usa o nome da classe para obter
a nota. Não troque nomes por `M1`–`M5`, pois isso perde a distinção entre as nove notas.

## Revisar e organizar imagens

A galeria de classificação tem abas Pendentes, Treino, Validação e Teste, filtro
por classe e páginas de 36 imagens. Clique na foto para ampliar e revisar a classe
ou mover para outra divisão. A seleção múltipla permanece entre páginas da mesma
divisão; “Selecionar esta página” adiciona as imagens visíveis (até 500 por operação).
Trocar divisão ou filtro limpa a seleção.

Os botões “Mover selecionadas” e “Excluir selecionadas” operam sobre a seleção.
Mover permite preservar a classe atual ou escolher outra. Imagens sem classe só
entram no treino/validação após receber uma classe. Nomes repetidos recebem um
sufixo ao mover/importar, preservando arquivos que já existem no destino.

Excluir pede confirmação e move os arquivos para
`projetos/<nome>/.lixeira_classificacao/<identificador>/`. Eles ficam fora do
dataset; para recuperar, copie os arquivos de volta à divisão e classe desejadas.
As alterações do dataset são bloqueadas durante o treinamento.

Se faltar validação para uma classe ativa, o treinador usa as imagens de treino
daquela classe também para conferência e avisa no log. Nesse caso as métricas
não representam uma validação independente. Para medir qualidade, adicione imagens
separadas em Validação. Pelo menos uma imagem em Treino é necessária para iniciar.
Cada execução registra as classes efetivamente usadas em 
`runs_treinamento/<treino>/classes_treinadas.json`.

O exportador mantém fotos do mesmo SIF/dia/sequencial na mesma divisão e aproxima a
porcentagem por classe. Revise também separação por animal/lote. Pastas sem imagens
de treino são ignoradas; não geram classes aprendidas no modelo.

Verificações locais: `.venv\Scripts\python.exe -m unittest test_treinamento -v`.
Os testes usam imagens sintéticas e um modelo simulado para conferir o despacho dos
três tipos de treino; não avaliam a precisão do modelo real.
