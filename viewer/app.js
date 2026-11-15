let data, cohort = 'comparison', selectedModel = null, trace = null, expanded = false;
const $ = selector => document.querySelector(selector);
const number = value => value == null ? '—' : value.toFixed(2);
const percent = value => value == null ? '—' : `${(value * 100).toFixed(1)}%`;
const text = (tag, value, className) => {const node=document.createElement(tag);node.textContent=value;if(className)node.className=className;return node;};
const clear = node => node.replaceChildren();
const formatStatus = status => ({FINISHED:'Completed',IN_PROGRESS:'In progress',EVALUATING:'Grading',ERROR:'Failed',STOPPED:'Stopped',AWAITING_UPDATE:'Awaiting local update',PENDING:'Starting'})[status] || status;
const harness = run => run.agent.startsWith('codex') ? 'Codex' : 'Claude Code';
function duration(value) {return value == null ? '—' : `${Math.floor(value/60)}m ${Math.round(value%60)}s`;}
function currentRuns() {return data.runs.filter(run=>run.cohort===cohort);}
async function refresh() {
  $('#refresh').disabled=true;
  try {const response=await fetch('/api/experiment');if(!response.ok)throw new Error('Unable to load saved results');data=await response.json();render();$('#loading').hidden=true;$('#update-note').textContent=`Read locally ${new Date(data.loaded_at).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}`;}
  catch(error){$('#loading').hidden=false;$('#loading').textContent=error.message;}
  finally{$('#refresh').disabled=false;}
}
function render() {
  const rows=data.leaderboard.filter(row=>row.cohort===cohort), runs=currentRuns();
  if(selectedModel && !rows.some(row=>row.model===selectedModel))selectedModel=null;
  $('#system-count').textContent=`(${rows.length})`;
  $('#completion').textContent=`${runs.filter(r=>r.score!=null).length} / ${runs.length} runs scored`;
  const body=$('#leaderboard tbody');clear(body);
  rows.forEach((row,index)=>{
    const tr=document.createElement('tr');if(row.model===selectedModel)tr.className='selected';
    const rank=text('td', row.mean==null?'—':String(rows.findIndex(other=>other.mean===row.mean)+1));tr.append(rank);
    const system=document.createElement('td'), name=text('div',row.label,'system');name.prepend(text('span','','dot'));name.append(text('small',`${harness(row)} · ${row.budget_seconds/60}-minute budget`));system.append(name);tr.append(system);
    tr.append(text('td',row.mean==null?'Pending':number(row.mean),`score${row.mean==null?' pending':''}`));
    const trialCell=document.createElement('td'), values=text('div','','trial-values');
    row.trials.forEach(id=>{const run=runs.find(r=>r.run_id===id);const pill=text('span',run.score==null?'…':number(run.score),`trial-pill${run.score==null?' pending':''}`);pill.title=`Trial ${run.repeat}: ${formatStatus(run.status)}`;values.append(pill);});
    trialCell.append(values,text('small',`${row.completed}/${row.total} scored`,'trial-count'));tr.append(trialCell);
    ['repair','specificity','transfer'].forEach(key=>tr.append(text('td',percent(row.diagnostics[key]),'diag')));
    tr.append(text('td',duration(row.duration),'diag'));
    const action=document.createElement('td'), button=text('button','View runs ↗');button.setAttribute('aria-label',`View ${row.label} runs`);action.append(button);tr.append(action);
    tr.addEventListener('click',()=>{selectedModel=selectedModel===row.model?null:row.model;render();$('#runs-section').scrollIntoView({behavior:'smooth',block:'start'});});body.append(tr);
  });
  const cards=$('#run-cards');clear(cards);
  const visible=runs.filter(run=>!selectedModel || run.model===selectedModel);
  $('#runs-title').textContent=selectedModel ? `${visible[0]?.label || ''} · individual runs`:'Individual runs';
  visible.forEach(run=>{
    const card=text('article','','run-card'), header=text('div','','run-header');
    header.append(text('h3',`${run.label} · trial ${run.repeat}`),text('span',formatStatus(run.status),`status${run.status==='FINISHED'?'':' pending'}`));
    const score=text('div',number(run.score),'run-score');score.append(text('small','net improvement / 100'));
    card.append(header,score,text('div',`${run.model}\n${run.agent} · reasoning ${run.reasoning_effort || 'not recorded'}\nAgent time ${duration(run.timings.agent_run_duration)} · budget ${run.budget_seconds/60}m`,'run-detail'));
    if(run.score!=null)card.append(text('div',`Repair ${percent(run.scores.repair)} · Specificity ${percent(run.scores.specificity)} · Transfer ${percent(run.scores.transfer)}`,'run-detail'));
    const button=text('button',run.has_trace?'Read unified trace ↗':'Trace not collected yet');button.disabled=!run.has_trace;button.addEventListener('click',()=>openTrace(run.run_id));card.append(button);cards.append(card);
  });
}
async function openTrace(id) {
  const dialog=$('#trace-dialog');if(!dialog.open)dialog.showModal();$('#trace-title').textContent='Loading trace…';clear($('#trace-meta'));clear($('#trace-body'));$('#trace-search').value='';expanded=false;$('#expand-all').textContent='Expand tools';
  try {const response=await fetch(`/api/trace/${encodeURIComponent(id)}`);if(!response.ok)throw new Error('Could not load this trace');trace=await response.json();const run=trace.run;$('#trace-title').textContent=`${run.label} · trial ${run.repeat} · ${run.budget_seconds/60} minutes`;
    const meta=$('#trace-meta');meta.append(text('span',run.model),text('span',`${harness(run)} ${trace.metadata?.cli_version || ''}`),text('strong',`Score ${number(run.score)} / 100`),text('span',`Agent ${duration(run.timings.agent_run_duration)}`));
    if(run.docent_url && run.docent_url.startsWith('https://docent.transluce.org/')){const link=text('a','Open in Docent ↗');link.href=run.docent_url;link.target='_blank';link.rel='noopener noreferrer';meta.append(link);}
    if(trace.patch){trace.messages.push({kind:'tool',label:'Submitted intervention',text:JSON.stringify(trace.patch,null,2),event_index:null});}
    $('#trace-note').textContent=trace.note;renderTrace();history.replaceState(null,'',`#run=${id}`);
  } catch(error){$('#trace-body').append(text('div',error.message,'empty'));}
}
function renderTrace() {
  const body=$('#trace-body');clear(body);const query=$('#trace-search').value.trim().toLowerCase();
  const messages=trace.messages.filter(event=>!query || `${event.label} ${event.text}`.toLowerCase().includes(query));
  $('#trace-count').textContent=`${messages.length} / ${trace.messages.length} events`;
  if(!messages.length){body.append(text('div','No matching events.','empty'));return;}
  const fragment=document.createDocumentFragment();
  messages.forEach(event=>{
    const section=text('article','',`event ${event.kind}`), header=text('div','','event-header');header.append(text('strong',event.label));if(event.timestamp)header.append(text('span',new Date(event.timestamp).toLocaleTimeString()));header.append(text('span',event.event_index==null?'Artifact':`#${event.event_index+1}`,'event-index'));section.append(header);
    if(event.kind==='message'){section.append(text('p',event.text,'event-text'));}
    else {const detail=document.createElement('details');detail.open=expanded || !!query;const preview=event.text.split('\n').find(line=>line.trim()) || '(empty result)';detail.append(text('summary',preview.slice(0,160)),text('pre',event.text));section.append(detail);}
    fragment.append(section);
  });body.append(fragment);
}
$('#refresh').addEventListener('click',refresh);
document.querySelectorAll('[data-cohort]').forEach(button=>button.addEventListener('click',()=>{cohort=button.dataset.cohort;selectedModel=null;document.querySelectorAll('[data-cohort]').forEach(tab=>tab.setAttribute('aria-selected',tab===button));render();}));
$('#close-trace').addEventListener('click',()=>$('#trace-dialog').close());
$('#trace-dialog').addEventListener('close',()=>history.replaceState(null,'',location.pathname));
$('#trace-search').addEventListener('input',()=>{if(trace)renderTrace();});
$('#expand-all').addEventListener('click',()=>{expanded=!expanded;$('#expand-all').textContent=expanded?'Collapse tools':'Expand tools';renderTrace();});
refresh().then(()=>{const id=new URLSearchParams(location.hash.slice(1)).get('run');if(id && data?.runs.some(run=>run.run_id===id))openTrace(id);});
