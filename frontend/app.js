const $ = s => document.querySelector(s);
const DEFAULT_API = "http://127.0.0.1:8000";
const apiBaseEl = $("#apiBase");
const docsLink = $("#docsLink");

if(apiBaseEl && docsLink) apiBaseEl.addEventListener("input", () => docsLink.href = apiBaseEl.value.replace(/\/$/, "") + "/docs");

function apiBase(){ return (apiBaseEl ? apiBaseEl.value : DEFAULT_API).replace(/\/$/, ""); }

async function checkHealth(){
  const dot = $("#apiDot"), txt = $("#apiText");
  if(!dot && !txt) {
    // status UI removed — just ping silently
    try{ await fetch(apiBase()+"/health"); }catch(e){ console.error("[health]", e); }
    return;
  }
  try{
    const r = await fetch(apiBase()+"/health");
    if(!r.ok) throw new Error(r.statusText);
    if(dot) dot.className="dot online";
    if(txt) txt.textContent="API connectée";
  }catch(e){
    if(dot) dot.className="dot offline";
    if(txt) txt.textContent="API déconnectée: "+e.message;
    console.error("[health]", e);
  }
}
checkHealth();

function showRaw(o){
  const sec = $("#rawJson");
  const pre = $("#rawJsonPre");
  if(sec) sec.classList.remove("hidden");
  if(pre) pre.textContent = JSON.stringify(o, null, 2);
}

function escapeHtml(s){
  if(!s) return "";
  return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
}

function showError(sectionId, msg){
  const sec = $(sectionId);
  if(!sec) return;
  sec.classList.remove("hidden");
  let banner = sec.querySelector(".error-banner");
  if(!banner){
    banner = document.createElement("div");
    banner.className = "error-banner";
    banner.style.cssText = "background:#fef2f2;border:1px solid #fecaca;color:#dc2626;padding:10px;border-radius:8px;margin-bottom:10px;white-space:pre-wrap;word-break:break-word;";
    sec.prepend(banner);
  }
  banner.textContent = msg;
  banner.style.display = "block";
}
function clearError(sectionId){
  const sec = $(sectionId);
  if(!sec) return;
  const b = sec.querySelector(".error-banner");
  if(b) b.style.display="none";
}

function safeChart(canvasId, config){
  const cvs = $(canvasId);
  if(!cvs) return null;
  if(typeof Chart === "undefined"){
    console.warn("Chart.js non chargé — graphique ignoré pour", canvasId);
    cvs.style.display="none";
    let warn = cvs.parentElement.querySelector(".chart-warn");
    if(!warn){
      warn = document.createElement("p");
      warn.className="chart-warn muted";
      warn.textContent="Chart.js non chargé (CDN bloqué / hors ligne) — tableaux restent disponibles.";
      cvs.insertAdjacentElement("afterend", warn);
    }
    return null;
  }
  cvs.style.display="";
  return new Chart(cvs, config);
}

let chunkChart, embedChart, retrievalChart;

function corpusPayload(){
  return {
    text: $("#corpus").value,
    chunk_size: parseInt($("#chunkSize").value||100),
    chunk_overlap: parseInt($("#chunkOverlap").value||20)
  };
}

async function fetchJson(url, opts){
  const r = await fetch(url, opts);
  const text = await r.text();
  let data;
  try{ data = JSON.parse(text); } catch{ data = {raw:text}; }
  if(!r.ok){
    const msg = data.error || data.detail || text.slice(0,400);
    throw new Error(`HTTP ${r.status} ${r.statusText}: ${msg}`);
  }
  return data;
}

const BTN_LABELS = {};
function setBtnLoading(id, loading){
  const btn = document.getElementById(id);
  if(!btn) return "";
  if(!(id in BTN_LABELS)) BTN_LABELS[id] = btn.innerHTML;
  if(loading){ btn.disabled = true; btn.innerHTML = '⏳ Chargement…'; }
  else{ btn.disabled = false; btn.innerHTML = BTN_LABELS[id]; }
  return BTN_LABELS[id];
}

async function runChunking(){
  if(document.getElementById("btnChunking")?.disabled) return;
  setBtnLoading("btnChunking", true);
  $("#secChunking").classList.remove("hidden");
  $("#chunkingCards").innerHTML = '<div class="muted">Chargement… (1er clic = chargement modèle SBERT, 10-30s)</div>';
  clearError("#secChunking");
  try{
    const payload = corpusPayload();
    const data = await fetchJson(apiBase()+"/chunking", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
    showRaw(data);
    const sec = $("#secChunking");
    sec.classList.remove("hidden");
    const grid = $("#chunkingCards");
    grid.innerHTML="";
    for(const [method,chunks] of Object.entries(data.results)){
      const card = document.createElement("div");
      card.className="method-card";
      card.innerHTML = `<h4>${escapeHtml(method)} <span class="muted">(${chunks.length} chunks)</span></h4>` +
        chunks.slice(0,4).map(c=>`<div class="chunk">${escapeHtml(c.slice(0,220))}${c.length>220?"…":""}</div>`).join("") +
        (chunks.length>4?`<div class="muted">+ ${chunks.length-4} autres…</div>`:"");
      grid.appendChild(card);
    }
  }catch(e){
    console.error("[runChunking]", e);
    showRaw({error:String(e), stack:e.stack});
    showError("#secChunking", "Erreur Chunking: "+e.message+" — Vérifie que le backend est lancé (http://127.0.0.1:8000/docs) et que l'URL API est correcte.");
  }finally{
    setBtnLoading("btnChunking", false);
  }
}

async function runChunkMetrics(){
  if(document.getElementById("btnChunkMetrics")?.disabled) return;
  setBtnLoading("btnChunkMetrics", true);
  $("#secChunkMetrics").classList.remove("hidden");
  const _tcb = $("#chunkMetricsTable tbody");
  if(_tcb) _tcb.innerHTML = '<tr><td colspan="6" class="muted">Calcul similarités… (peut prendre 10-20s la 1ère fois)</td></tr>';
  clearError("#secChunkMetrics");
  try{
    const payload = corpusPayload();
    const data = await fetchJson(apiBase()+"/chunking/metrics", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
    showRaw(data);
    $("#secChunkMetrics").classList.remove("hidden");
    const tbody = $("#chunkMetricsTable tbody");
    tbody.innerHTML="";
    const labels=[], vals=[];
    for(const [m,s] of Object.entries(data.metrics)){
      labels.push(m); vals.push(s.intra_similarity);
      const tr=document.createElement("tr");
      tr.innerHTML=`<td><b>${escapeHtml(m)}</b></td><td>${s.nb_chunks}</td><td>${s.longueur_moyenne}</td><td>${s.ecart_type}</td><td>${s.intra_similarity}</td><td class="muted">${escapeHtml((s.exemple||"").slice(0,90))}</td>`;
      tbody.appendChild(tr);
    }
    $("#bestChunking").textContent = `Meilleure méthode : ${data.best_method}  (${data.best_chunks.length} chunks)  — critère : intra_similarity − 0.01·nb_chunks`;
    if(window._cc) try{window._cc.destroy();}catch{}
    window._cc = safeChart("#chunkChart", {
      type:"bar",
      data:{labels, datasets:[{label:"Cohérence intra", data:vals, backgroundColor:"#0d9488"}]},
      options:{responsive:true, plugins:{legend:{display:false}}, scales:{y:{min:0,max:1}}}
    });
  }catch(e){
    console.error("[runChunkMetrics]", e);
    showRaw({error:String(e)});
    showError("#secChunkMetrics", "Erreur Métriques Chunking: "+e.message);
  }finally{
    setBtnLoading("btnChunkMetrics", false);
  }
}

async function runEmbedding(){
  if(document.getElementById("btnEmbedding")?.disabled) return;
  setBtnLoading("btnEmbedding", true);
  $("#secEmbedding").classList.remove("hidden");
  const _etb0 = $("#embedMetricsTable tbody");
  if(_etb0) _etb0.innerHTML = '<tr><td colspan="5" class="muted">Calcul embeddings… (glove/bert exclus par défaut car très lents — voir console)</td></tr>';
  clearError("#secEmbedding");
  try{
    // light methods only: glove (66MB download) + bert (440MB) freeze the UI for minutes
    const body = { text: $("#corpus").value, methods: ["tfidf", "word2vec", "fasttext", "doc2vec", "sentence_bert"] };
    const data = await fetchJson(apiBase()+"/embedding", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
    showRaw(data);
    $("#secEmbedding").classList.remove("hidden");
    const tbody = $("#embedMetricsTable tbody");
    tbody.innerHTML="";
    const labels=[], vals=[];
    for(const [m,info] of Object.entries(data.embedding_metrics)){
      const ok = info.intra_similarity!==undefined;
      if(ok){ labels.push(m); vals.push(info.intra_similarity); }
      const tr=document.createElement("tr");
      tr.innerHTML=`<td><b>${escapeHtml(m)}</b></td><td>${info.dim??"-"}</td><td>${info.nb_vectors??"-"}</td><td>${info.intra_similarity??"-"}</td><td>${info.error?`<span style="color:#dc2626">${escapeHtml(info.error.slice(0,120))}</span>`:"OK"}</td>`;
      tbody.appendChild(tr);
    }
    $("#bestEmbedding").textContent = data.best_embedding_method ? `Meilleure méthode d'embedding : ${data.best_embedding_method}` : "Aucune méthode valide";
    const grid = $("#embeddingCards");
    grid.innerHTML = `<div class="method-card"><h4>Best chunks (${escapeHtml(data.best_chunking_method)})</h4>` +
      data.best_chunks.map(c=>`<div class="chunk">${escapeHtml(c.slice(0,180))}</div>`).join("") + `</div>`;
    if(window._ec) try{window._ec.destroy();}catch{}
    window._ec = safeChart("#embedChart", {
      type:"bar",
      data:{labels, datasets:[{label:"Intra-similarity", data:vals, backgroundColor:"#16a34a"}]},
      options:{responsive:true, plugins:{legend:{display:false}}, scales:{y:{min:0,max:1}}}
    });
    return data;
  }catch(e){
    console.error("[runEmbedding]", e);
    showRaw({error:String(e)});
    // ensure section visible with error so user sees something
    $("#secEmbedding").classList.remove("hidden");
    const tbody = $("#embedMetricsTable tbody");
    if(tbody) tbody.innerHTML = `<tr><td colspan="5" style="color:#dc2626">Erreur: ${escapeHtml(e.message)}</td></tr>`;
    showError("#secEmbedding", "Erreur Embedding: "+e.message+" — Vérifie le backend /embedding (voir console F12 > Network).");
    throw e; // rethrow for caller to know
  }finally{
    setBtnLoading("btnEmbedding", false);
  }
}

const LIGHT_RETRIEVAL = ["similarite_brute", "faiss", "bm25", "hybride"];
// dpr + reranking excluded by default: they download ~80-300MB on first call and freeze UI

async function runRetrieval(){
  $("#secRetrieval").classList.remove("hidden");
  const _rtb0 = $("#retrievalTable tbody");
  if(_rtb0) _rtb0.innerHTML = '<tr><td colspan="5" class="muted">Recherche… (dpr/reranking exclus par défaut car très lents)</td></tr>';
  clearError("#secRetrieval");
  try{
    const k = parseInt($("#k").value||3);
    const base = { query: $("#query").value, text: $("#corpus").value, k };
    // one fast call per light method — avoids one slow method (dpr/rerank) blocking everything
    const merged = { query: base.query, retrieval: {} };
    let raw = null;
    for(const m of LIGHT_RETRIEVAL){
      const d = await fetchJson(apiBase()+"/retrieval", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({...base, method: m})});
      raw = raw || d;
      merged.best_chunks = d.best_chunks;
      merged.relevant_texts = d.relevant_texts;
      Object.assign(merged.retrieval, d.retrieval);
    }
    const data = merged;
    showRaw({...data, note: "dpr + reranking exclus (téléchargements lourds). Appelle POST /retrieval {method:'dpr'} directement si besoin."});
    $("#secRetrieval").classList.remove("hidden");
    const grid = $("#retrievalCards");
    grid.innerHTML="";
    const tbody = $("#retrievalTable tbody");
    tbody.innerHTML="";
    const labels=[], pVals=[], rVals=[], mrrVals=[];
    for(const [m,obj] of Object.entries(data.retrieval)){
      if(obj.error){
        grid.innerHTML += `<div class="method-card"><h4>${escapeHtml(m)}</h4><div style="color:#dc2626">${escapeHtml(obj.error.slice(0,200))}</div></div>`;
        continue;
      }
      labels.push(m); pVals.push(obj.metrics["precision@k"]); rVals.push(obj.metrics["recall@k"]); mrrVals.push(obj.metrics.mrr);
      grid.innerHTML += `<div class="method-card"><h4>${escapeHtml(m)}</h4>` +
        obj.results.map(x=>`<div class="chunk"><b>${x.score}</b> — ${escapeHtml(x.chunk.slice(0,140))}</div>`).join("") + `</div>`;
      const tr=document.createElement("tr");
      tr.innerHTML=`<td><b>${escapeHtml(m)}</b></td><td>${obj.metrics["precision@k"]}</td><td>${obj.metrics["recall@k"]}</td><td>${obj.metrics.mrr}</td><td class="muted">${escapeHtml((obj.results[0]?.chunk||"").slice(0,70))}</td>`;
      tbody.appendChild(tr);
    }
    if(window._rc) try{window._rc.destroy();}catch{}
    window._rc = safeChart("#retrievalChart", {
      type:"bar",
      data:{labels, datasets:[
        {label:"Precision@k", data:pVals, backgroundColor:"#1e40af"},
        {label:"Recall@k", data:rVals, backgroundColor:"#f59e0b"},
        {label:"MRR", data:mrrVals, backgroundColor:"#64748b"},
      ]},
      options:{responsive:true, scales:{y:{min:0,max:1}}}
    });
    return data;
  }catch(e){
    console.error("[runRetrieval]", e);
    showRaw({error:String(e)});
    $("#secRetrieval").classList.remove("hidden");
    const tbody = $("#retrievalTable tbody");
    if(tbody) tbody.innerHTML = `<tr><td colspan="5" style="color:#dc2626">Erreur: ${escapeHtml(e.message)}</td></tr>`;
    showError("#secRetrieval", "Erreur Retrieval: "+e.message);
    throw e;
  }
}

async function runEmbedMetrics(){
  const btn = document.getElementById("btnEmbedMetrics");
  if(btn?.disabled) return;
  const origText = BTN_LABELS["btnEmbedMetrics"] || (btn ? btn.innerHTML : "");
  BTN_LABELS["btnEmbedMetrics"] = origText;
  if(btn){ btn.disabled=true; btn.innerHTML='<span class="icon">⏳</span> Chargement…'; }
  // ensure sections visible immediately so user sees something happens
  $("#secEmbedding").classList.remove("hidden");
  $("#secRetrieval").classList.remove("hidden");
  clearError("#secEmbedding");
  clearError("#secRetrieval");
  // Optional pre-fill loading placeholders
  const etb = $("#embedMetricsTable tbody");
  if(etb) etb.innerHTML = `<tr><td colspan="5" class="muted">Chargement 7 embeddings… (peut prendre 10-20s la 1ère fois)</td></tr>`;
  const rtb = $("#retrievalTable tbody");
  if(rtb) rtb.innerHTML = `<tr><td colspan="5" class="muted">Chargement 6 retrievals…</td></tr>`;

  let embErr = null, retErr = null;
  try{ await runEmbedding(); } catch(e){ embErr = e; }
  try{ await runRetrieval(); } catch(e){ retErr = e; }

  if(btn){ btn.disabled=false; btn.innerHTML=origText; }

  if(embErr && retErr){
    showRaw({error:"Embed+Retrieval both failed", embed:String(embErr), retrieval:String(retErr)});
  } else if(embErr){
    console.warn("Embedding failed but retrieval succeeded", embErr);
  } else if(retErr){
    console.warn("Retrieval failed but embedding succeeded", retErr);
  }
  // scroll to results
  try{ $("#secEmbedding").scrollIntoView({behavior:"smooth", block:"start"});}catch{}
}

async function runVectorial(){
  try{
    const q = encodeURIComponent($("#query").value);
    const k = $("#k").value;
    const data = await fetchJson(apiBase()+`/vectorial?query=${q}&k=${k}`);
    showAnswer(data);
    showRaw(data);
  }catch(e){
    console.error("[runVectorial]", e);
    showRaw({error:String(e)});
    alert("Erreur Vectorial: "+e.message);
  }
}

async function runQuery(){
  try{
    const body = { query: $("#query").value, text: $("#corpus").value, k: parseInt($("#k").value||3), retrieval_method: $("#retrievalMethod").value };
    const data = await fetchJson(apiBase()+"/query", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
    showAnswer(data);
    showRaw(data);
  }catch(e){
    console.error("[runQuery]", e);
    showRaw({error:String(e)});
    alert("Erreur Query: "+e.message);
  }
}

function showAnswer(data){
  if(data.error){ alert(data.error); return; }
  $("#secAnswer").classList.remove("hidden");
  $("#generatedResponse").textContent = data.generated_response || "—";
  $("#answerNote").textContent = data.note || "";
  $("#contextRaw").textContent = data.context || "";
  const ol = $("#retrievedChunks");
  ol.innerHTML="";
  (data.retrieved_chunks||[]).forEach(x=>{
    const li=document.createElement("li");
    li.textContent = `[${x.score}] ${x.chunk}`;
    ol.appendChild(li);
  });
}

function drawBar(id, ref, labels, vals, label){
  // helper unused — charts are created inline via safeChart
}

// Fallback wiring: ensure buttons work even if inline onclick is stripped by CSP/cache
document.addEventListener("DOMContentLoaded", () => {
  const bind = (id, fn) => { const b=document.getElementById(id); if(b && !b.dataset.bound){ b.addEventListener("click", fn); b.dataset.bound="1"; }};
  bind("btnChunking", runChunking);
  bind("btnChunkMetrics", runChunkMetrics);
  bind("btnEmbedding", runEmbedding);
  bind("btnEmbedMetrics", runEmbedMetrics);
  // expose globally for inline onclick
  window.runChunking = runChunking;
  window.runChunkMetrics = runChunkMetrics;
  window.runEmbedding = runEmbedding;
  window.runEmbedMetrics = runEmbedMetrics;
  window.runVectorial = runVectorial;
  window.runQuery = runQuery;
  window.checkHealth = checkHealth;
  console.log("[RAG] Frontend ready — 4 boutons câblés. API=", apiBase());
});
window.addEventListener("error", e=>{ console.error("[window.onerror]", e.message, e.error); });
window.addEventListener("unhandledrejection", e=>{ console.error("[unhandledrejection]", e.reason); });
