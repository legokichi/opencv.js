importScripts(["cv.js"]);
Module.addOnPostRun(()=>{
  self.postMessage(true);
  console.log("OnPostRun");

  const cascade = new Module.CascadeClassifier();
  if(! cascade.load("./data/haarcascades/haarcascade_frontalface_default.xml") ){
    throw new Error("cannot load cascade file");
  }

  self.onmessage = ({data})=>{
    console.log("onmessage");
    const imgData = data;

    const mat = Module.matFromArray(imgData, 24); // 24 for rgba

    const mat_gray = new Module.Mat();
    Module.cvtColor(mat, mat_gray, Module.ColorConversionCodes.COLOR_RGBA2GRAY.value, 0);

    const faces = new Module.RectVector();
    cascade.detectMultiScale(mat_gray, faces, 1.1, 3, 0, [0, 0], [0, 0]);

    const rects = [];
    for (let i=0; i<faces.size(); i+=1){
      const faceRect = faces.get(i);
      x = faceRect.x ;
      y = faceRect.y ;
      w = faceRect.width ;
      h = faceRect.height;
      rects[i] = {x, y, w, h};
      faceRect.delete();
    }

    faces.delete();
    mat.delete();
    mat_gray.delete();

    self.postMessage(rects);
  };
});