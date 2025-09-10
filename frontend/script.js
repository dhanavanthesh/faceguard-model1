// frontend/script.js
const socket = io();

let video = document.getElementById("video");
let overlay = document.getElementById("overlay");
let ctx = overlay.getContext("2d");
let startBtn = document.getElementById("startBtn");
let stopBtn = document.getElementById("stopBtn");
let downloadBtn = document.getElementById("downloadLog");
let logTableBody = document.querySelector("#logTable tbody");
let detectionNameEl = document.getElementById("detectionName");
let detectionConfEl = document.getElementById("detectionConfidence");
let signExamples = document.getElementById("signExamples");

let capturing = false;
let sendInterval = null;
let log = [];

// Sample sign images & short descriptions (you can replace with real images)
const SIGNS = [
  {id:"closed_fist", title:"Closed Fist", desc:"All fingers curled into fist. Neutral/hold gesture."},
  {id:"open_palm", title:"Open Palm", desc:"All fingers extended, palm facing camera. Used for 'Stop'."},
  {id:"pointing_up", title:"Pointing Up", desc:"Index finger extended up; others folded."},
  {id:"thumbs_down", title:"Thumbs Down", desc:"Thumb pointing downward."},
  {id:"thumbs_up", title:"Thumbs Up", desc:"Thumb pointing upward (OK/affirmative)."},
  {id:"victory", title:"Victory", desc:"Index and middle extended forming V (peace)."},
  {id:"i_love_you", title:"I Love You", desc:"Extended thumb and pinky; other fingers folded."},
  {id:"left_hand_up", title:"Left Hand Up", desc:"Left hand raised above shoulder."},
  {id:"right_hand_up", title:"Right Hand Up", desc:"Right hand raised above shoulder."},
  {id:"stop", title:"Stop", desc:"Open palm facing camera."},
  {id:"help", title:"Help (Both Thumbs Up)", desc:"Show thumbs up with both hands."},
  {id:"sos", title:"SOS (Crossed Arms)", desc:"Cross both forearms across your chest."},
  {id:"fire_alarm", title:"Fire Alarm (Hands Up - Y Shape)", desc:"Raise both hands above head with open palms (Y-shape)."},
  {id:"chest_pain", title:"Chest Pain", desc:"Hand near chest/center region."}
];

function renderSignSamples() {
  signExamples.innerHTML = "";
  SIGNS.forEach(s=>{
    const d = document.createElement("div");
    d.className = "sign-sample";
    d.innerHTML = `<strong>${s.title}</strong><div style="font-size:11px;color:#9fb6c5">${s.desc}</div>`;
    signExamples.appendChild(d);
  });
}
renderSignSamples();

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({video:{width:640,height:480}, audio:false});
  video.srcObject = stream;
  await video.play();
  // set canvas size considering CSS size and devicePixelRatio so drawings align crisply
  const dpr = window.devicePixelRatio || 1;
  const rect = video.getBoundingClientRect();
  overlay.style.width = rect.width + "px";
  overlay.style.height = rect.height + "px";
  overlay.width = Math.round(rect.width * dpr);
  overlay.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function sendFrame() {
  if (!capturing) return;
  // draw video to an offscreen canvas and get jpeg
  const off = document.createElement("canvas");
  off.width = video.videoWidth;
  off.height = video.videoHeight;
  const offCtx = off.getContext("2d");
  offCtx.drawImage(video, 0, 0, off.width, off.height);
  const dataUrl = off.toDataURL("image/jpeg", 0.6);
  socket.emit("frame", {image: dataUrl, ts: Date.now()});
}

socket.on("connect", ()=> console.log("socket connected"));
socket.on("server_ready", (m)=> console.log(m));

socket.on("detection", (payload) => {
  // recompute DPR transform and clear in device pixels for crisp alignment
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(1,0,0,1,0,0);
  ctx.clearRect(0,0,overlay.width, overlay.height);
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.lineWidth = 4;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  // Map model coordinates (frame_size) to canvas when video uses object-fit: cover
  let scaleX = 1, scaleY = 1, dx = 0, dy = 0;
  if (payload.frame_size && video.videoWidth && video.videoHeight) {
    const srcW = payload.frame_size.width;
    const srcH = payload.frame_size.height;
    const rect = video.getBoundingClientRect();
    const dstW = rect.width;
    const dstH = rect.height;
    const scale = Math.max(dstW / srcW, dstH / srcH); // cover
    const drawW = srcW * scale;
    const drawH = srcH * scale;
    dx = (dstW - drawW) / 2;
    dy = (dstH - drawH) / 2;
    scaleX = scaleY = scale;
  }
  const mapX = (x)=> x * scaleX + dx;
  const mapY = (y)=> y * scaleY + dy;
  // pose skeleton
  if (payload.skeleton_lines) {
    ctx.strokeStyle = "rgba(6,182,212,0.9)";
    payload.skeleton_lines.forEach(l=>{
      ctx.beginPath();
      ctx.moveTo(mapX(l[0]), mapY(l[1]));
      ctx.lineTo(mapX(l[2]), mapY(l[3]));
      ctx.stroke();
    });
  }
  // pose joints
  if (payload.pose_points) {
    ctx.fillStyle = "rgba(6,182,212,0.95)";
    payload.pose_points.forEach(p=>{
      const r = 3.5;
      ctx.beginPath();
      ctx.arc(mapX(p[0]), mapY(p[1]), r, 0, Math.PI*2);
      ctx.fill();
    });
  }
  // left hand
  if (payload.left_hand_lines) {
    ctx.strokeStyle = "rgba(255,120,120,0.95)";
    payload.left_hand_lines.forEach(l=>{
      ctx.beginPath();
      ctx.moveTo(mapX(l[0]), mapY(l[1]));
      ctx.lineTo(mapX(l[2]), mapY(l[3]));
      ctx.stroke();
    });
  }
  if (payload.left_hand_points) {
    ctx.fillStyle = "rgba(255,120,120,0.95)";
    payload.left_hand_points.forEach(p=>{
      const r = 3.5;
      ctx.beginPath();
      ctx.arc(mapX(p[0]), mapY(p[1]), r, 0, Math.PI*2);
      ctx.fill();
    });
  }
  // right hand
  if (payload.right_hand_lines) {
    ctx.strokeStyle = "rgba(120,255,140,0.95)";
    payload.right_hand_lines.forEach(l=>{
      ctx.beginPath();
      ctx.moveTo(mapX(l[0]), mapY(l[1]));
      ctx.lineTo(mapX(l[2]), mapY(l[3]));
      ctx.stroke();
    });
  }
  if (payload.right_hand_points) {
    ctx.fillStyle = "rgba(120,255,140,0.95)";
    payload.right_hand_points.forEach(p=>{
      const r = 3.5;
      ctx.beginPath();
      ctx.arc(mapX(p[0]), mapY(p[1]), r, 0, Math.PI*2);
      ctx.fill();
    });
  }

  // determine highest-confidence detection
  let detectionText = "—";
  let confidenceText = "—";
  if (payload.detections && payload.detections.length>0) {
    const best = payload.detections.sort((a,b)=>b.confidence - a.confidence)[0];
    detectionText = best.name;
    confidenceText = (best.confidence*100).toFixed(1) + "%";
    // log it
    const now = new Date().toLocaleString();
    log.unshift({time:now,event:best.name,confidence:best.confidence});
    updateLogTable();
  } else {
    // fallback to hand gestures
    if (payload.hand_gestures) {
      if (payload.hand_gestures.left && payload.hand_gestures.left.gesture !== "none") {
        detectionText = "Left: " + payload.hand_gestures.left.gesture;
        confidenceText = (payload.hand_gestures.left.confidence*100).toFixed(1) + "%";
      }
      if (payload.hand_gestures.right && payload.hand_gestures.right.gesture !== "none") {
        detectionText = "Right: " + payload.hand_gestures.right.gesture;
        confidenceText = (payload.hand_gestures.right.confidence*100).toFixed(1) + "%";
      }
    }
  }
  detectionNameEl.textContent = detectionText;
  detectionConfEl.textContent = "Confidence: " + confidenceText;
});

socket.on("error", (e) => {
  console.error("server error:", e);
});

function updateLogTable(){
  logTableBody.innerHTML = "";
  for(let i=0;i<Math.min(log.length,100);i++){
    const r = log[i];
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${r.time}</td><td>${r.event}</td><td>${(r.confidence*100).toFixed(1)}%</td>`;
    logTableBody.appendChild(tr);
  }
}

startBtn.addEventListener("click", async ()=>{
  await startCamera();
  capturing = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  // send frames periodically (20 FPS cap)
  sendInterval = setInterval(sendFrame, 100); // 10 fps in practice
});

stopBtn.addEventListener("click", ()=>{
  capturing = false;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  clearInterval(sendInterval);
  // stop tracks
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(t=>t.stop());
    video.srcObject = null;
  }
  ctx.clearRect(0,0,overlay.width,overlay.height);
});

downloadBtn.addEventListener("click", ()=>{
  // generate CSV
  let csv = "time,event,confidence\n";
  for (const r of log) {
    csv += `"${r.time}","${r.event}",${r.confidence}\n`;
  }
  const blob = new Blob([csv], {type:"text/csv"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "detection_log.csv"; document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
});
