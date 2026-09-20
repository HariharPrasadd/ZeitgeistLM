"""Serve a live local loss chart from the active Modal training call's logs."""

import argparse
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import threading
import time

import modal


TRAIN = re.compile(r"step (\d+) loss ([\d.]+)")
VALIDATION = re.compile(r"validation step (\d+) loss ([\d.]+)")


class LossStore:
    """Incrementally fetch log lines and keep one value per training step."""

    def __init__(self, call_id: str):
        self.call = modal.FunctionCall.from_id(call_id)
        self.train = {}
        self.validation = {}
        self.since = None
        self.error = None
        self.lock = threading.Lock()

    def update(self):
        try:
            # Overlap by one second because many steps can share a log timestamp.
            since = self.since - timedelta(seconds=1) if self.since else None
            newest = self.since
            training, validation = {}, {}
            for entry in self.call.logs.fetch(since=since, source="stdout"):
                newest = max(newest, entry.timestamp) if newest else entry.timestamp
                for line in entry.message.splitlines():
                    match = TRAIN.fullmatch(line.split(" lr ")[0]) if " lr " in line else None
                    if match:
                        training[int(match.group(1))] = float(match.group(2))
                    match = VALIDATION.fullmatch(line)
                    if match:
                        validation[int(match.group(1))] = float(match.group(2))
            with self.lock:
                self.train.update(training)
                self.validation.update(validation)
                self.since = newest
                self.error = None
        except Exception as exc:
            with self.lock:
                self.error = str(exc)

    def snapshot(self):
        with self.lock:
            return {
                "train": sorted(self.train.items()),
                "validation": sorted(self.validation.items()),
                "error": self.error,
            }


PAGE = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ZeitgeistLM loss · rollback run</title>
<style>
body{margin:0;background:#101318;color:#eceff3;font:16px system-ui,sans-serif}
main{max-width:1100px;margin:38px auto;padding:0 24px}
h1{font-size:28px;margin:0 0 8px}p{color:#aeb9c6;margin:0 0 24px}
.stats{display:flex;gap:30px;margin:0 0 20px}.stats strong{display:block;font-size:25px;color:#fff}
.stats span{color:#aeb9c6;font-size:13px}canvas{width:100%;height:480px;background:#191f27;border-radius:12px}
.legend{margin-top:14px;color:#aeb9c6}.train{color:#62d6b0}.val{color:#f5bc66}
#error{color:#f08080;margin-top:12px}
</style><main><h1>ZeitgeistLM training loss</h1><p>Rollback run from the 1B-token checkpoint · refreshes every 15 seconds</p>
<div class="stats"><div><strong id="step">—</strong><span>Latest step</span></div>
<div><strong id="training">—</strong><span>Training loss</span></div>
<div><strong id="validation">—</strong><span>Last 2021 validation loss</span></div></div>
<canvas id="chart"></canvas><div class="legend"><span class="train">● Training</span> &nbsp; <span class="val">● Validation</span></div>
<div id="error"></div></main><script>
const canvas=document.getElementById('chart'),ctx=canvas.getContext('2d');
function draw(data){
 const w=canvas.clientWidth,h=canvas.clientHeight,dpr=devicePixelRatio||1;
 canvas.width=w*dpr;canvas.height=h*dpr;ctx.scale(dpr,dpr);
 ctx.fillStyle='#191f27';ctx.fillRect(0,0,w,h);
 const train=data.train,val=data.validation;
 if(!train.length)return;
 const left=55,right=w-22,top=22,bottom=h-38,maxStep=train.at(-1)[0];
 const all=[...train,...val].map(p=>p[1]);
 // Ignore the first few high-loss warmup points when scaling the visible chart.
 const visible=train.length>100?all.filter(v=>v<=Math.max(5,train[100][1])):all;
 const minY=Math.max(0,Math.min(...visible)-.15),maxY=Math.max(...visible)+.15;
 const x=s=>left+(right-left)*s/Math.max(1,maxStep),y=v=>bottom-(bottom-top)*(v-minY)/(maxY-minY);
 ctx.strokeStyle='#33404d';ctx.fillStyle='#aeb9c6';ctx.font='12px system-ui';ctx.lineWidth=1;
 for(let i=0;i<=4;i++){let value=minY+(maxY-minY)*i/4,py=y(value);
  ctx.beginPath();ctx.moveTo(left,py);ctx.lineTo(right,py);ctx.stroke();ctx.fillText(value.toFixed(2),8,py+4)}
 for(let i=0;i<=4;i++){let step=Math.round(maxStep*i/4),px=x(step);
  ctx.fillText(step.toLocaleString(),px-12,h-13)}
 function line(points,color,width){ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=width;
  let started=false;for(const [step,loss] of points){if(loss>maxY||loss<minY)continue;
   if(!started){ctx.moveTo(x(step),y(loss));started=true}else ctx.lineTo(x(step),y(loss))}ctx.stroke()}
 line(train,'#62d6b0',1.5);line(val,'#f5bc66',2.5);
 document.getElementById('step').textContent=maxStep.toLocaleString();
 document.getElementById('training').textContent=train.at(-1)[1].toFixed(4);
 document.getElementById('validation').textContent=val.length?val.at(-1)[1].toFixed(4):'Pending';
}
let latest=null;async function refresh(){try{const response=await fetch('/metrics');latest=await response.json();
 draw(latest);document.getElementById('error').textContent=latest.error||''}
 catch(error){document.getElementById('error').textContent=String(error)}}
addEventListener('resize',()=>{if(latest)draw(latest)});refresh();setInterval(refresh,15000);
</script></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Keep the live chart on the current rollback run; the original is archived in loss/.
    parser.add_argument("--call-id", default="fc-01M2YW89TG9YECYWS12TZ957TF")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    store = LossStore(args.call_id)

    def poll():
        while True:
            store.update()
            time.sleep(15)

    threading.Thread(target=poll, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/metrics":
                content, mime = json.dumps(store.snapshot()).encode(), "application/json"
            elif self.path == "/":
                content, mime = PAGE.encode(), "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Loss dashboard: http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
