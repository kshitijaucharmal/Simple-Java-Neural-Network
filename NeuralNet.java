class NeuralNet{
  int inodes, hnodes, onodes;
  float lr;
  
  float[][] wih, who;
  float[][] one;
  
  NeuralNet(int inodes, int hnodes, int onodes, float lr){
    this.inodes = inodes;
    this.hnodes = hnodes;
    this.onodes = onodes;
    this.lr = lr;
    
    wih = new float[hnodes][inodes];
    who = new float[onodes][hnodes];
    initWeights(wih);
    initWeights(who);
  }
  
  void mutateWeights(){
    for(int i = 0; i < who.length; i++){
      for(int j = 0; j < who[0].length; j++){
        if(random(1) < 0.5)
          who[i][j] = random(-1, 1);
      }
    }
    for(int i = 0; i < wih.length; i++){
      for(int j = 0; j < wih[0].length; j++){
        if(random(1) < 0.5)
          wih[i][j] = random(-1, 1);
      }
    }
  }
  
  float[][] createOne(int a, int b){
    float[][] x = new float[a][b];
    for(int i = 0; i < x.length; i++){
      for(int j = 0; j < x[0].length; j++){
        x[i][j] = 1.0;
      }
    }
    return x;
  }
  
  void initWeights(float[][] x){
    for(int i = 0; i < x.length; i++){
      for(int j = 0; j < x[0].length; j++){
        x[i][j] = random(-1, 1);
      }
    }
  }
  
  float[][] query(float[][] inputs){
    inputs = T(inputs);
    float[][] hidden_outputs = sigmoid(Dot(wih, inputs));
    float[][] final_outputs = sigmoid(Dot(who, hidden_outputs));
    return final_outputs;
  }
  
  void train(float[][] inputs, float[][] targets){
    inputs = T(inputs);
    targets = T(targets);
    
    float[][] hidden_outputs = sigmoid(Dot(wih, inputs));
    float[][] final_outputs = sigmoid(Dot(who, hidden_outputs));
    
    float[][] final_errors = calc(targets, final_outputs, 's');
    float[][] hidden_errors = Dot(T(who), final_errors);
    
    float[][] temp = calc(createOne(final_outputs.length, final_outputs[0].length), final_outputs, 's');
    temp = calc(final_outputs, temp, 'm');
    temp = calc(final_errors, temp, 'm');
    temp = Dot(temp, T(hidden_outputs));
    temp = m(temp, lr);
    
    who = calc(who, temp, 'a');
    
    temp = calc(createOne(hidden_outputs.length, hidden_outputs[0].length), hidden_outputs, 's');
    temp = calc(hidden_outputs, temp, 'm');
    temp = calc(hidden_errors, temp, 'm');
    temp = Dot(temp, T(inputs));
    temp = m(temp, lr);
    
    wih = calc(wih, temp, 'a');
  }
  
  float[][] m(float[][] x, float l){
    float[][] s = new float[x.length][x[0].length];
    for(int i = 0; i < x.length; i++){
      for(int j = 0; j < x[0].length; j++){
        s[i][j] = x[i][j] * l;
      }
    }
    return s;
  }
  
  void printa(float[][] x){
    for(int i = 0; i < x.length; i++){
      for(int j = 0; j < x[0].length; j++) print(x[i][j] + " ");
      println();
    }
    println("\n\n");
  }
  
  float[][] Dot(float[][] a, float[][] b){
    if(a[0].length != b.length){
      print("Matrix mult not possible");
      return new float[][]{{2}};
    }
    float[][] s = new float[a.length][b[0].length];
    for(int i = 0; i < a.length; i++){
      for(int j = 0; j < b[0].length; j++){
        for(int k = 0; k < b.length; k++) s[i][j] += a[i][k] * b[k][j];
      }
    }
    return s;
  }
  
  float[][] sigmoid(float[][] x){
    float[][] s = new float[x.length][x[0].length];
    for(int i = 0; i < x.length; i++){
      for(int j = 0; j < x[0].length; j++) s[i][j] = 1 / (1 + exp(-x[i][j]));
    }
    return s;
  }
  float[][] th(float[][] x){
    float[][] s = new float[x.length][x[0].length];
    for(int i = 0; i < x.length; i++){
      for(int j = 0; j < x[0].length; j++) s[i][j] = (float)Math.tanh(-x[i][j]);
    }
    return s;
  }
  
  float[][] calc(float[][] x, float[][] y, char d){
    float[][] s = new float[x.length][x[0].length];
    for(int i = 0; i < x.length; i++){
      for(int j = 0; j < x[0].length; j++){
        switch(d){
          case 'a':
            s[i][j] = x[i][j] + y[i][j];
            break;
          case 's':
            s[i][j] = x[i][j] - y[i][j];
            break;
          case 'm':
            s[i][j] = x[i][j] * y[i][j];
            break;
          case 'd':
            s[i][j] = x[i][j] / y[i][j];
            break;
          default:
            println("Invalid");
            return s;
        }
      }
    }
    return s;
  }
  
  float[][] T(float[][] a){
    float[][]s = new float[a[0].length][a.length];
    for(int i = 0; i < a.length; i++){
      for(int j = 0; j < a[0].length; j++) s[j][i] = a[i][j];
    }
    return s;
  }
}
