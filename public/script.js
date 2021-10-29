// Load our saved model from current directory (which will be
// hosted via Firebase Hosting)
async function predict() {
  // Relative URL provided for my-model.json.
  const model = await tf.loadLayersModel("./my-model.json");
  // Once model is loaded, let's try using it to make a prediction!
  // Print to developer console for now.
  const results = model.predict(tf.tensor2d([[1.3, 0.3, 0.5]])).toString();
  console.log(JSON.stringify(results));
}

predict();
