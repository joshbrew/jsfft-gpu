# jsfft-gpu
GPU DFT implementation. It implements a straightforward discrete fourier transform on the GPU with gpu.js. Performance is much faster than jsfft and scales incredibly well.

Based on this awesome DFT guide: https://github.com/trekhleb/javascript-algorithms/tree/master/src/algorithms/math/fourier-transform

## Benchmark on RTX 2060 using gpuworker.js:

##### 128 channels, 512sps, 1 second of data with bandpass filter: 8ms fastest, 20ms average.

##### 1 channel, 512sps, 1 second of data: 1.9ms, 2.5ms average

## Important functions

`var gpu = new gpuUtils()`

`gpu.gpuDFT(signalBuffer, nSeconds, texOut = false)` - Single channel DFT, give it a 1D array with the number of seconds of data captured and it will return a 2D array, the first index is a list of the frequency distribution (x-axis) of each amplitude, the second index is the array of amplitudes up to a max of the sample rate / 2 (the nyquist frequency of the sampling rate).

`gpu.MultiChannelDFT(signalBuffer, nSeconds, texOut = false)` - Multi channel DFT, input a 2D array with rows of equal widths and the number of seconds sampled. Outputs a 2D array with the frequency distribution (x-axis) in the first index and then the list of amplitudes in the second index.

`gpu.MultiChannelDFT_BandPass(signalBuffer, nSeconds, freqStart, freqEnd, texOut = false)` - Multi channel DFT with a bandpass filter applied. Input 2D array with rows of equal widths, number of seconds, high pass frequency (lower bound), and low pass frequency (upper bound). Just set the filter to maximum nyquist sampling frequency for no low-pass filtering. Outputs a 2D array with the band pass window frequency distribution in the first index and the list of -positive- amplitudes in the next index.

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

gpu.addFunction(function mag(a,b){ // Returns magnitude
        return Math.sqrt(a*a + b*b);
});

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

## GPUJS multichannel DFT
```
var gpu = new GPU();

      gpu.addFunction(function mag(a,b){ // Returns magnitude
        return Math.sqrt(a*a + b*b);
      });


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
      var listdft1D = gpu.createKernel(function(signals,len){
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


//128 channel test with bandpass.
            var sigList1D = [...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave,...sineWave];
            //console.log(sigList1D);
            console.time("gpuListDFT");
            listdft1D.setOutput([sigList1D.length]); //Set output to length of list of signals
            listdft1D.setLoopMaxIterations(sineWave.length); //Set loop size to the length of one signal (assuming all are uniform length
            gpuClass.listdft1D_windowed.setOutput([sigList1D.length]); //Set output to length of list of signals
            gpuClass.listdft1D_windowed.setLoopMaxIterations(sineWave.length); //Set loop size to the length of one signal (assuming all are uniform length)
            //var gpuresult2 = gpuClass.listdft2D(sigList2D);
            //var gpuresult3 = gpuClass.listdft1D(sigList1D,sineWave.length/sec);
            freqStart = 0;
            freqEnd_nyquist = 50*2; //2x the end frequency of our bandpass (nyquist sampling rate)

            var gpuresult3 = gpuClass.listdft1D_windowed(sigList1D,sineWave.length/sec,freqStart,freqEnd_nyquist);
            console.timeEnd("gpuListDFT (128 channels)");
            //console.log(gpuresult3)
            orderedMagsList = [];
            posMagsList = [];
            for(var i = 0; i < gpuresult3.length; i+=sineWave.length){
                if(i < sineWave.length / sec) {
                    posMagsList.push([...gpuresult3.slice(i,Math.ceil(sineWave.length*.5+i))]);
                    orderedMagsList.push([...gpuresult3.slice(Math.ceil(sineWave.length*.5+i),sineWave.length+i),...gpuresult3.slice(i,Math.ceil(sineWave.length*.5+i))]);
                }
            }

            var summedMags = [];
            //console.log(posMagsList);
            if(sec > 1) { //Need to sum results when sample time > 1 sec
                //for(var k = 0; k < 128; k++){
                    //console.log(k)
                    for(var i = 0; i < posMagsList[0].length; i++ ){
                        if(i < fs){
                            summedMags.push(posMagsList[0][i]);
                            //if (i === 0 ) { console.log(summedMags[0]); }
                        }
                        else {
                            var j = i-fs*Math.floor(i/fs);
                            summedMags[j] = (summedMags[j] * posMagsList[0][i-1]/posMagsList[0].length);
                            //if (j === 0 ) { console.log(summedMags[j]); }
                        }

                    }
                    summedMags = [...summedMags.slice(0,Math.ceil(summedMags.length*0.5))]
                //}
            }
            else {summedMags = posMagsList[0];}
            
            //console.log(summedMags.length);

            var fftwindow = [];
            for (i = 0; i < Math.ceil(0.5*sineWave.length/sec); i++){
                fftwindow.push(freqStart + (freqEnd_nyquist-freqStart)*i/(sineWave.length/sec));
            }
            
```
           
