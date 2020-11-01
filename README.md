# jsfft-gpu
 Trying to translate jsfft for gpujs, it's confusing. Working very simplistic DFT example using series notation, plus a hacky but ultra fast FFT compared to jsfft, just increase the seconds of simulated data to show how the GPU performance stays fairly stable while CPU time skyrockets for the DFT example, but there are a lot of optimizations possible I will be trying. Check the console for timer results.

## 30Hz with 150Hz interference added simulation.
![fftsnip](fftsnip.PNG)


## CPU simple DFT code
```
console.time("simpledft");
var real = [];
var imag = [];
var mags = [];
var TWOPI = 2*3.141592653589793
for(var k=0; k<sineWave.length;k++){
    real.push(0);
    imag.push(0);
    for(var j=0;j<sineWave.length;j++){
        var shared = TWOPI*k*j/sineWave.length
        real[k] = real[k]+sineWave[j]*Math.cos(shared);
        imag[k] = imag[k]-sineWave[j]*Math.sin(shared);
    }
    mags.push(Math.sqrt(real[k]*real[k]+imag[k]*imag[k]));
}
//console.log(mags);
console.timeEnd("simpledft");
```

## GPUJS simple DFT code
```
//in head: <script src=gpu-browser.min.js charset="UTF-8"></script>
var gpu = new GPU();
gpu.addFunction(function DFT(signal,len,freq){ //Extract a particular frequency
var real = 0;
var imag = 0;
for(var i = 0; i<len; i++){
  var shared = 6.28318530718*freq*i/len; //this.thread.x is the target frequency
  real = real+signal[i]*Math.cos(shared);
  imag = imag-signal[i]*Math.sin(shared);
}
//var mag = Math.sqrt(real[k]*real[k]+imag[k]*imag[k]);
return [real,imag]; //mag(real,imag)
});

//Return frequency domain based on DFT
var dft = gpu.createKernel(function (signal,len){
  var result = DFT(signal,len,this.thread.x);
  return mag(result[0],result[1]);
})
.setDynamicOutput(true)
.setDynamicArguments(true);
      
console.time("gpuDFT");
dft.setOutput([sineWave.length]);
dft.setLoopMaxIterations(sineWave.length);
var gpuresult = dft(sineWave,sineWave.length);
console.timeEnd("gpuDFT");

//Order the magnitudes by frequency
var orderedMags = [...gpuresult.slice(Math.ceil(gpuresult.length/2),gpuresult.length),...gpuresult.slice(0,Math.ceil(gpuresult.length/2))];
       
```

## GPUJS multichannel DFT, I got 40ms for 128channels and 512 samples per channel on page refresh. Not bad!!!
```
var gpu = new GPU();
      gpu.addFunction(function DFT(signal,len,freq){ //Extract a particular frequency
        var real = 0;
        var imag = 0;
        for(var i = 0; i<len; i++){
          var shared = 6.28318530718*freq*i/len; //this.thread.x is the target frequency
          real = real+signal[i]*Math.cos(shared);
          imag = imag-signal[i]*Math.sin(shared);
        }
        //var mag = Math.sqrt(real[k]*real[k]+imag[k]*imag[k]);
        return [real,imag]; //mag(real,imag)
      });

      gpu.addFunction(function DFTlist(signals,len,freq,n){ //Extract a particular frequency
        var real = 0;
        var imag = 0;
        for(var i = 0; i<len; i++){
          var shared = 6.28318530718*freq*i/len; //this.thread.x is the target frequency
          real = real+signals[i+len*n]*Math.cos(shared);
          imag = imag-signals[i+len*n]*Math.sin(shared);
        }
        //var mag = Math.sqrt(real[k]*real[k]+imag[k]*imag[k]);
        return [real,imag]; //mag(real,imag)
      });
      
            //More like a vertex buffer list to chunk through lists of signals
      var listdft1D = this.gpu.createKernel(function(signals,len){
        var result = [0,0];
        if(this.thread.x <= len){
          result = DFT(signals,len,this.thread.x);
        }
        else{
          var n = Math.floor(this.thread.x/len);
          result = DFTlist(signals,len,this.thread.x-n*len,n);
        }
        return mag(result[0],result[1]);
      })
      .setDynamicOutput(true)
      .setDynamicArguments(true)


//128 channel test, 1 second of data @ 512sps
            var sigList1D = [...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave];
            //console.log(sigList1D);
            console.time("gpuListDFT");
            listdft1D.setOutput([sigList1D.length]); //Set output to length of list of signals
            listdft1D.setLoopMaxIterations(sineWave.length); //Set loop size to the length of one signal (assuming all are uniform length
            var gpuresult3 = gpuClass.listdft1D(sigList1D,sineWave.length);

            console.timeEnd("gpuListDFT");
        ;
            
            orderedMagsList = [];
            for(var i = 0; i < gpuresult3.length; i+=sineWave.length){
                //console.log(i);
                orderedMagsList.push([...gpuresult3.slice(Math.ceil(sineWave.length*.5+i),sineWave.length+i),...gpuresult3.slice(i,Math.ceil(sineWave.length*.5+i))]);
            }
            
```
           
