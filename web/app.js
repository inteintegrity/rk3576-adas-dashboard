const $=id=>document.getElementById(id);
let current=null, timer=null, polling=false;
const number=v=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(1):'—';
async function api(path,options){const r=await fetch(path,options);if(!r.ok){let m;try{m=(await r.json()).detail}catch{m=r.statusText}throw new Error(m)}return r.json()}
function busy(value){$('submit').disabled=value;$('sample').disabled=value;}
function render(job){
  const m=job.metrics||{}, complete=job.status==='completed';
  $('status').textContent=job.status.charAt(0).toUpperCase()+job.status.slice(1);
  $('filename').textContent=job.filename;
  $('npu').textContent=number(m.npu_fps);$('fps').textContent=number(m.processing_fps);
  $('frames').textContent=m.frames??'—';$('fcw').textContent=m.fcw_frames??'—';$('ldw').textContent=m.ldw_frames??'—';
  const total=m.total_frames||m.source_frames||0;
  $('duration').textContent=total?`${total} source frames`:'Waiting for analysis';
  $('bar').style.width=complete?'100%':`${total?Math.min(100,100*(m.frames||0)/total):0}%`;
  $('progress-text').textContent=complete?'Analysis complete':job.status==='failed'?'Analysis failed':`${m.frames||0} / ${total||'—'} frames`;
  $('downloads').hidden=!complete;
  if(complete){
    $('empty').hidden=true;$('preview').hidden=true;$('video').hidden=false;
    const path=`/api/jobs/${job.id}/video`;
    if($('video').getAttribute('src')!==path){$('video').src=path;$('video').load()}
    $('download').href=path+'?download=true';$('report').href=`/api/jobs/${job.id}/report?download=true`;
    $('trace').href=`/api/jobs/${job.id}/trace?download=true`;
  }else if(m.frames){
    $('empty').hidden=true;$('video').hidden=true;$('preview').hidden=false;
    $('preview').src=`/api/jobs/${job.id}/preview?t=${Date.now()}`;
  }
  if(job.status==='failed')$('message').textContent=job.error||'Analysis failed';
}
async function refreshHistory(){
  const list=await api('/api/jobs');$('history').replaceChildren(new Option('Select a completed run',''));
  for(const job of list.filter(j=>j.status==='completed'))$('history').add(new Option(`${job.filename} · ${number(job.metrics.processing_fps)} FPS`,job.id));
  if(current)$('history').value=current;
  return list;
}
async function poll(){if(!current||polling)return;polling=true;
  try{const job=await api(`/api/jobs/${current}`);render(job);
    if(['completed','failed'].includes(job.status)){clearInterval(timer);busy(false);await refreshHistory()}
  }catch(e){$('message').textContent=e.message}finally{polling=false}
}
async function select(id){clearInterval(timer);current=id;$('message').textContent='';await poll();
  const job=await api(`/api/jobs/${id}`);if(!['completed','failed'].includes(job.status)){busy(true);timer=setInterval(poll,1000)}
}
async function start(path,options){busy(true);$('message').textContent='Preparing video…';
  try{const job=await api(path,options);$('message').textContent='';$('video').pause();$('video').hidden=true;$('preview').hidden=true;$('empty').hidden=false;await select(job.id)}
  catch(e){$('message').textContent=e.message;busy(false)}
}
$('file').onchange=()=>{$('selected').textContent=$('file').files[0]?.name||'Up to 512 MB'};
$('form').onsubmit=e=>{e.preventDefault();const file=$('file').files[0];if(!file)return;if(file.size>512*1024*1024){$('message').textContent='Please choose an MP4 smaller than 512 MB.';return}start('/api/jobs',{method:'POST',body:new FormData($('form'))})};
$('sample').onclick=()=>start('/api/jobs/sample',{method:'POST'});
$('history').onchange=()=>{if($('history').value)select($('history').value)};
(async()=>{try{const health=await api('/api/health');$('connection').textContent=health.ready?'RK3576 NPU connected':'NPU unavailable';
  const list=await refreshHistory();const running=list.find(j=>['processing','uploading'].includes(j.status));
  const job=running||list.find(j=>j.status==='completed'&&j.id==='rk3576-final')||list.find(j=>j.status==='completed');
  if(job)await select(job.id);
}catch(e){$('connection').textContent='Device unavailable';$('message').textContent=e.message;busy(true)}})();
