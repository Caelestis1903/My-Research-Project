# Tested C04 publication byte reader, reusable for C05

Use this JavaScript inside functions.exec with its existing tools binding. It reads exact base64 in bounded tool chunks. Pass the return directly to the connected GitHub create_blob tool; do not print base64. Every promise is awaited and each result checked. Parent continues polling RAM owner independently at most60 seconds apart. Keep upload batches under40 seconds; no independent/background AI, shell session or driver is needed.

```javascript

const shellQuote=s=>"'"+s.replaceAll("'","'\"'\"'")+"'";
async function readB64(path,size){
 const block=768000,parts=new Array(Math.ceil(size/block));
 for(let start=0;start<parts.length;start+=16){
  const batch=Array.from({length:Math.min(16,parts.length-start)},(_,k)=>start+k);
  const rs=await Promise.allSettled(batch.map(i=>tools.exec_command({cmd:"dd if="+shellQuote(path)+" bs="+block+" skip="+i+" count=1 status=none | base64 -w0",max_output_tokens:300000,yield_time_ms:1000})));
  for(let k=0;k<rs.length;k++){
   const r=rs[k],i=batch[k];if(r.status!=="fulfilled"||r.value.exit_code!==0)throw Error("read block failed "+i);
   const out=r.value.output.trim(),expected=4*Math.ceil(Math.min(block,size-i*block)/3);
   if(out.length!==expected||!/^[A-Za-z0-9+/]*={0,2}$/.test(out))throw Error("truncated base64 "+i+" "+out.length+"/"+expected);
   parts[i]=out;
  }
 }
 const all=parts.join("");if(all.length!==4*Math.ceil(size/3))throw Error("base64 length");
 return all;
}

```

