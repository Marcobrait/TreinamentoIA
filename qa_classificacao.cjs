// Prévia isolada: execute test_treinamento.py --preview antes deste script.
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawn } = require('node:child_process');
const assert = require('node:assert/strict');
const pause = ms => new Promise(resolve => setTimeout(resolve,ms));
const temp = fs.mkdtempSync(path.join(os.tmpdir(),'classify-browser-'));
const browser = spawn('C:/Program Files/Google/Chrome/Application/chrome.exe',[
  '--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',
  '--remote-debugging-port=5098','--user-data-dir='+temp,'about:blank'
],{windowsHide:true,stdio:'ignore'});
let ws;
async function main() {
  let targets;
  for (let n=0;n<50;n++) {
    try { targets=await (await fetch('http://127.0.0.1:5098/json')).json(); if (targets.length) break; } catch {}
    await pause(100);
  }
  ws = new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);
  await new Promise(resolve=>ws.addEventListener('open',resolve,{once:true}));
  let id=0; const pending=new Map(), errors=[];
  ws.addEventListener('message',ev=>{
    const m=JSON.parse(ev.data);
    if (m.id) {
      const p=pending.get(m.id); pending.delete(m.id);
      if (m.error) p.reject(new Error(JSON.stringify(m.error))); else p.resolve(m.result);
    }
    if (m.method==='Runtime.exceptionThrown') errors.push(JSON.stringify(m.params.exceptionDetails));
  });
  const call=(method,params={})=>new Promise((resolve,reject)=>{
    const key=++id; pending.set(key,{resolve,reject}); ws.send(JSON.stringify({id:key,method,params}));
  });
  const evaluate=async expression=>{
    const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});
    if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails));
    return r.result.value;
  };
  const wait=async expression=>{
    for (let n=0;n<100;n++) { if (await evaluate(expression)) return; await pause(100); }
    throw new Error('Tempo excedido: '+expression);
  };
  await call('Runtime.enable');
  await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
  await call('Page.navigate',{url:'http://127.0.0.1:5099'});
  await wait("document.querySelectorAll('#class-galeria .class-card input').length === 36");
  await evaluate("document.getElementById('class-galeria').scrollIntoView({block:'center'})");
  await wait("[...document.querySelectorAll('#class-galeria img')].slice(0,6).every(i=>i.complete && i.naturalWidth>0)");
  const desktop=path.join(temp,'desktop.png');
  fs.writeFileSync(desktop,Buffer.from((await call('Page.captureScreenshot')).data,'base64'));
  const status=await evaluate("({text:document.getElementById('class-galeria-info').textContent,classes:document.getElementById('class-preview-classe').options.length})");
  assert.match(status.text,/54/); assert.equal(status.classes,10);
  await evaluate("document.getElementById('class-selecionar-pagina').click()");
  assert.equal(await evaluate("classState.selecionadas.size"),36);
  await evaluate("document.querySelector('#class-paginacao button:last-child').click()");
  await wait("classState.pagina===2 && document.querySelectorAll('#class-galeria input').length===18 && !classState.carregando");
  assert.equal(await evaluate("classState.selecionadas.size"),36);
  await evaluate("document.querySelector('#class-galeria .class-photo').click()");
  assert.equal(await evaluate("document.getElementById('class-preview').open"),true);
  await evaluate("document.getElementById('class-preview-fechar').click(); document.getElementById('class-limpar-selecao').click()");
  await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
  await evaluate("document.getElementById('class-abas').scrollIntoView({block:'start'})");
  const mobile=path.join(temp,'mobile.png');
  fs.writeFileSync(mobile,Buffer.from((await call('Page.captureScreenshot')).data,'base64'));
  const overflow=await evaluate("document.documentElement.scrollWidth > window.innerWidth");
  assert.equal(overflow,false,'Overflow no celular');
  await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
  await evaluate("document.getElementById('classificacao-painel').scrollIntoView({block:'start'})");
  await evaluate("(async()=>{ const response=await fetch('/classificacao/imagem?projeto=MarmoreioClassify&chave=val/1/foto-1-val.png'); const blob=await response.blob(); const dt=new DataTransfer(); dt.items.add(new File([blob],'avulsa.png',{type:'image/png'})); document.getElementById('class-arquivos').files=dt.files; document.getElementById('form-class-imagens').requestSubmit(); })()");
  await wait("classState.busy && !document.getElementById('class-upload-status').hidden");
  assert.ok(await evaluate("document.querySelectorAll('#class-galeria .class-skeleton').length>0"));
  const skeleton=path.join(temp,'skeleton.png');
  fs.writeFileSync(skeleton,Buffer.from((await call('Page.captureScreenshot')).data,'base64'));
  await wait("!classState.busy && classState.pasta==='pendentes' && !classState.carregando && classState.itens.length===1");
  await evaluate("document.getElementById('class-selecionar-pagina').click(); document.getElementById('class-mover-classe').value='9'; document.getElementById('class-mover-destino').value='validacao'; document.getElementById('class-mover').click()");
  await wait("!classState.busy && !classState.carregando && classState.itens.length===0");
  assert.equal(errors.length,0,errors.join('\n'));
  console.log(JSON.stringify({ok:true,desktop,mobile,skeleton,status},null,2));
  await call('Browser.close').catch(()=>{});
}
main().catch(err=>{console.error(err);process.exitCode=1;}).finally(()=>{
  if (ws) ws.close();
  browser.kill();
  // Perfil e capturas sintéticas ficam no diretório temporário informado.
});
