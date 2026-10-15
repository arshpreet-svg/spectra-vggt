import * as THREE from 'three';
import {OrbitControls} from './vendor/OrbitControls.js';
import {PLYLoader} from './vendor/PLYLoader.js';
const $ = id => document.getElementById(id);
let currentJob, cloud, cameraPath, bounds;
const scene = new THREE.Scene();
scene.background = new THREE.Color('#131c16');
const camera = new THREE.PerspectiveCamera(55, 1, 0.001, 10000);
// OpenCV world orientation: display +Y down consistently with camera convention.
camera.up.set(0,-1,0);
let renderer;
try {
  renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  $('viewer').appendChild(renderer.domElement);
} catch(error) {
  $('status').textContent = `WebGL unavailable: ${error.message}`;
  throw error;
}
const controls = new OrbitControls(camera,renderer.domElement);
controls.enableDamping = true;
const resize = () => {
  const {width,height} = $('viewer').getBoundingClientRect();
  renderer.setSize(width,height); camera.aspect=width/height; camera.updateProjectionMatrix();
};
new ResizeObserver(resize).observe($('viewer'));
function fit() {
  if(!bounds) return;
  const center = bounds.getCenter(new THREE.Vector3());
  const radius = Math.max(bounds.getSize(new THREE.Vector3()).length()/2, .01);
  const distance = radius / Math.sin(THREE.MathUtils.degToRad(camera.fov/2)) * Math.max(1,1/camera.aspect) * 1.2;
  camera.position.copy(center).add(new THREE.Vector3(distance*.6,-distance*.4,-distance));
  camera.near=radius/1000; camera.far=Math.max(distance*100,100); camera.updateProjectionMatrix();
  controls.target.copy(center); controls.update();
}
$('reset').onclick=fit;
$('size').oninput=()=>{if(cloud) cloud.material.size=Number($('size').value);};
renderer.setAnimationLoop(()=>{controls.update(); renderer.render(scene,camera);});
async function json(url, options) {
  const response=await fetch(url,options);
  const data=await response.json();
  if(!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
function status(message,error=false){$('status').textContent=message; $('dot').style.background=error?'#f39484':'#b5f5ce';}
async function loadJob(id,meta){
  const geometry=await new PLYLoader().loadAsync(`/api/jobs/${id}/cloud.ply`);
  if(cloud){scene.remove(cloud);cloud.geometry.dispose();cloud.material.dispose();}
  if(cameraPath){scene.remove(cameraPath);cameraPath.geometry.dispose();cameraPath.material.dispose();}
  const material=new THREE.PointsMaterial({size:Number($('size').value),sizeAttenuation:false,vertexColors:true});
  cloud=new THREE.Points(geometry,material); scene.add(cloud);
  geometry.computeBoundingBox();bounds=geometry.boundingBox.clone();
  const centers=meta.cameras.map(item=>new THREE.Vector3(...item.center));
  if(centers.length>1){cameraPath=new THREE.Line(new THREE.BufferGeometry().setFromPoints(centers),new THREE.LineBasicMaterial({color:0xb5f5ce}));scene.add(cameraPath);}
  fit();$('empty').hidden=true;
  $('scene-title').textContent=`SCENE / ${id.slice(0,8).toUpperCase()}`;
  $('engine').textContent=`VGGT · ${meta.runtime.backend || meta.runtime.device || 'PyTorch'}`;
  $('points').textContent=meta.point_count.toLocaleString();$('views').textContent=meta.cameras.length;
  $('timing').textContent=`${meta.inference_seconds.toFixed(2)} s`;
  for(const [element,file] of [['ply','cloud.ply'],['json','metadata.json'],['numpy','predictions.npz']]) $(''+element).href=`/api/jobs/${id}/${file}?download=1`;
  $('downloads').hidden=false; $('events').disabled=false;
  currentJob=id;localStorage.setItem('spectra-last-job',id);
  await refreshInspection();
}
$('files').onchange=()=>{$('selection').textContent=Array.from($('files').files).map(f=>f.name).join(', ')||'No files selected';};
$('upload').onsubmit=async event=>{
  event.preventDefault();$('run').disabled=true;status('Uploading scene…');
  try {
    const job=await json('/api/reconstruct',{method:'POST',body:new FormData($('upload'))});
    let result;
    do {
      await new Promise(resolve=>setTimeout(resolve,1500));
      result=await json(`/api/jobs/${job.id}`);
      status(result.status==='running'?'VGGT is reconstructing · first run may download model weights…':result.status);
    } while(['queued','running'].includes(result.status));
    if(result.status!=='complete') throw new Error(result.error || 'Reconstruction interrupted');
    await loadJob(job.id,result.metadata);status('Reconstruction complete · PLY verified');
  } catch(error){status(error.message,true);}finally{$('run').disabled=false;}
};
$('events').onclick=async()=>{
  try {$('detections').textContent=JSON.stringify(await json(`/api/jobs/${currentJob}/detections`),null,2);}
  catch(error){status(error.message,true);}
};
try{
 const health=await json('/api/health');
 $('runtime').textContent=`${health.config.backend.toUpperCase()} · ${health.cuda_available?'CUDA available':'CPU runtime'}`;
 $('limits').textContent=`Up to ${health.config.max_frames} images · ${health.config.max_upload_mb} MB maximum`;
 const last=localStorage.getItem('spectra-last-job');
 if(last){const job=await json(`/api/jobs/${last}`);if(job.status==='complete'){await loadJob(last,job.metadata);status('Previous reconstruction restored');}}
}catch(error){status(error.message,true);}

async function refreshInspection(){
 if(!currentJob) return;
 try {const report=await json(`/api/jobs/${currentJob}/inspection`);$('advisory').textContent=`${report.action.replaceAll('_',' ').toUpperCase()}: ${report.reason}`;
 $('advisory').style.color=report.action==='pause_scan'?'#f39484':'#9fbaa9';}
 catch(error){$('advisory').textContent=error.message;}
}
setInterval(refreshInspection,3000);
