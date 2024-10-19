#ifdef GL_ES
precision mediump float;
#endif

// Phong related variables
uniform sampler2D uSampler;
uniform vec3 uKd;
uniform vec3 uKs;
uniform vec3 uLightPos;
uniform vec3 uCameraPos;
uniform vec3 uLightIntensity;

varying highp vec2 vTextureCoord;
varying highp vec3 vFragPos;
varying highp vec3 vNormal;

// Shadow map related variables
#define NUM_SAMPLES 20
#define BLOCKER_SEARCH_NUM_SAMPLES NUM_SAMPLES
#define PCF_NUM_SAMPLES NUM_SAMPLES
#define NUM_RINGS 10

#define EPS 1e-3
#define PI 3.141592653589793
#define PI2 6.283185307179586
#define LIGHT_WORLD_SIZE 5. // 光源大小比例值
#define NEAR_PLANE 1e-2 // 光源近平面
#define FAR_PLANE 500. // 光源远平面大小
#define LIGHT_SIZE_UV (LIGHT_WORLD_SIZE / FAR_PLANE) // 纹理中对应的大小

// #define NEAR_PLANE 0.01
// #define LIGHT_WORLD_SIZE 5.
// #define LIGHT_SIZE_UV LIGHT_WORLD_SIZE / FRUSTUM_SIZE

uniform sampler2D uShadowMap;

varying vec4 vPositionFromLight;
varying highp vec3 shadowCoord;
highp float rand_1to1(highp float x){
  // -1 -1
  return fract(sin(x)*10000.);
}

highp float rand_2to1(vec2 uv){
  // 0 - 1
  const highp float a=12.9898,b=78.233,c=43758.5453;
  highp float dt=dot(uv.xy,vec2(a,b)),sn=mod(dt,PI);
  return fract(sin(sn)*c);
}

float unpack(vec4 rgbaDepth){
  const vec4 bitShift=vec4(1.,1./256.,1./(256.*256.),1./(256.*256.*256.));
  return dot(rgbaDepth,bitShift);
}

vec2 poissonDisk[NUM_SAMPLES];

void poissonDiskSamples(const in vec2 randomSeed){
  
  float ANGLE_STEP=PI2*float(NUM_RINGS)/float(NUM_SAMPLES);
  float INV_NUM_SAMPLES=1./float(NUM_SAMPLES);
  
  float angle=rand_2to1(randomSeed)*PI2;
  float radius=INV_NUM_SAMPLES;
  float radiusStep=radius;
  
  for(int i=0;i<NUM_SAMPLES;i++){
    poissonDisk[i]=vec2(cos(angle),sin(angle))*pow(radius,.75);
    radius+=radiusStep;
    angle+=ANGLE_STEP;
  }
}

void uniformDiskSamples(const in vec2 randomSeed){
  
  float randNum=rand_2to1(randomSeed);
  float sampleX=rand_1to1(randNum);
  float sampleY=rand_1to1(sampleX);
  
  float angle=sampleX*PI2;
  float radius=sqrt(sampleY);
  
  for(int i=0;i<NUM_SAMPLES;i++){
    poissonDisk[i]=vec2(radius*cos(angle),radius*sin(angle));
    
    sampleX=rand_1to1(sampleY);
    sampleY=rand_1to1(sampleX);
    
    angle=sampleX*PI2;
    radius=sqrt(sampleY);
  }
}

// float findBlocker(sampler2D shadowMap,vec2 uv,float zReceiver){
//   return 1.;
// }
// float findBlocker( sampler2D shadowMap,  vec2 uv, float zReceiver ) {
//   int blocknum=0;
//   float blockdepth=0.0;
//   float lightdepth_z=vPositionFromLight.z;

//   //以光源为摄像机，通过将ShadowMap放到近平面上来计算每次采样的范围
//   //光源采样方式为投影，得在世界空间里计算
//   float lightSize=5.0;
//   float near_plane=0.01;

//   float filterRadius=(lightSize/400.0)*(lightdepth_z-near_plane)/lightdepth_z;

//   poissonDiskSamples(uv);
//   for(int i=0;i<NUM_SAMPLES;i++)
//   {
//     float dBlocker=unpack(texture2D(shadowMap,uv+poissonDisk[i]*filterRadius));
//     //计算深度在坐标范围在(0,1)的处理后的NDC空间里操作
//     if(dBlocker<zReceiver)
//     {
//       blocknum++;
//       blockdepth+=dBlocker;
//     }
//   }
//   return blockdepth/float(blocknum);
// }
float findBlocker( sampler2D shadowMap,  vec2 uv, float zReceiver ) {
    poissonDiskSamples(uv);
  int blockNum = 0;
  float blockDepth = 0.0, regionSize = LIGHT_SIZE_UV * (zReceiver - NEAR_PLANE) / zReceiver;
  for(int i = 0; i < BLOCKER_SEARCH_NUM_SAMPLES; ++ i) {
    float depth = unpack(texture2D(shadowMap, uv + poissonDisk[i] * regionSize));
    if(depth + EPS <= zReceiver) {
      blockNum ++;
      blockDepth += depth;
    }
  }
  if(blockNum == 0) 
    return -1.0;
  return blockDepth / float(blockNum);
}
// float PCF(sampler2D shadowMap,vec4 coords){
  
//   return 1.;
// }

float PCSS(sampler2D shadowMap,vec4 coords){
  
  // STEP 1: avgblocker depth
  
  // STEP 2: penumbra size
  
  // STEP 3: filtering
  
  return 1.;
  
}


float getShadowBias(float c, float filterRadiusUV){
  vec3 normal = normalize(vNormal);
  vec3 lightDir = normalize(uLightPos - vFragPos);
  float A = (1. + ceil(filterRadiusUV)) * (400.0 / 2048.0 / 2.0);
  return max(A, A * (1.0 - dot(normal, lightDir))) * c;
}

float useShadowMap(sampler2D shadowMap,vec4 shadowCoord,float biasC, float filterRadiusUV){
  //将shadowmap转换为具体的数值,确保精度
  float curdepth=shadowCoord.z;
  float targetdepth=unpack(texture2D(shadowMap,shadowCoord.xy));
  float bias = getShadowBias(biasC, filterRadiusUV);
  //处于阴影中
  if(curdepth-bias>=targetdepth+EPS)
  {
    return 0.;
  } 
  else
  {
    return 1.;
  }
  
}
// float PCF(sampler2D shadowMap,vec4 coords,float biasC, float filterRadiusUV){
//     poissonDiskSamples(coords.xy); //这里用的是随机生成序列，通过coords.xy随机生成数
//   float visibility = 0.0;
//   for(int i = 0; i < NUM_SAMPLES; i++){
//     vec2 offset = poissonDisk[i] * filterRadiusUV;
//     float noshadow = useShadowMap(shadowMap, coords + vec4(offset, 0., 0.), biasC, filterRadiusUV);
//     if(noshadow!=0.0){
//       visibility++;
//     }
//   }
//   return visibility / float(NUM_SAMPLES);
//   // return 1.;
// }
float PCF(sampler2D shadowMap,vec4 coords,float biasC, float filterRadiusUV){
  float currentDepth = coords.z;
  poissonDiskSamples(coords.xy); // possion采样
  // uniformDiskSamples(coords.xy); // 均匀采样
  float visibility = 0.0;
  for(int i = 0; i < NUM_SAMPLES; ++ i) {
    float closestDepth = unpack(texture2D(shadowMap, poissonDisk[i] * filterRadiusUV + coords.xy)); 
    if (currentDepth <= closestDepth + EPS) {
      visibility += 1.0;
    }
  }
  visibility /= float(NUM_SAMPLES); // 加权平均
  return visibility;
  // return 1.;
}
float PCSS(sampler2D shadowMap, vec4 coords,float biasC){
  float zReceiver=coords.z;
  float avgDepth=findBlocker(shadowMap,coords.xy,zReceiver);

  
  float penumbra=(zReceiver-avgDepth)*(5.0/500.0)/avgDepth;
  float filterRadiusUV=penumbra;

  return PCF(shadowMap,coords,biasC,filterRadiusUV);
}

vec3 blinnPhong(){
  vec3 color=texture2D(uSampler,vTextureCoord).rgb;
  color=pow(color,vec3(2.2));
  
  vec3 ambient=.05*color;
  
  vec3 lightDir=normalize(uLightPos);
  vec3 normal=normalize(vNormal);
  float diff=max(dot(lightDir,normal),0.);
  vec3 light_atten_coff=
  uLightIntensity/pow(length(uLightPos-vFragPos),2.);
  vec3 diffuse=diff*light_atten_coff*color;
  
  vec3 viewDir=normalize(uCameraPos-vFragPos);
  vec3 halfDir=normalize((lightDir+viewDir));
  float spec=pow(max(dot(halfDir,normal),0.),32.);
  vec3 specular=uKs*light_atten_coff*spec;
  
  vec3 radiance=(ambient+diffuse+specular);
  vec3 phongColor=pow(radiance,vec3(1./2.2));
  return phongColor;
}

void main(void){
  //shadowmap
  // float visibility;
  // // vec3 shadowCoord=vPositionFromLight.xyz;
  // vec3 shadowCoord=vPositionFromLight.xyz/vPositionFromLight.w;
  // shadowCoord.xyz = (shadowCoord.xyz + 1.0) / 2.0;
  // float bias = 0.1;
  // visibility=useShadowMap(uShadowMap,vec4(shadowCoord,1.) ,bias, 0.);
  // visibility = PCF(uShadowMap, vec4(shadowCoord, 1.0,bias, 0.));
  // //visibility = PCSS(uShadowMap, vec4(shadowCoord, 1.0));
  


  //pcf

  //  float visibility;
  // float bias = 0.1;
  //  float filterRadiusUV = 10.0 / 2048.0;
  // vec3 shadowCoord=vPositionFromLight.xyz/vPositionFromLight.w;
  // shadowCoord.xyz=(shadowCoord.xyz+1.0)/2.0;
  // //visibility = useShadowMap(uShadowMap, vec4(shadowCoord, 1.0), bias, 0.0);
  // visibility = PCF(uShadowMap, vec4(shadowCoord, 1.0), bias, filterRadiusUV);

  //pcss
  //   float visibility = 1.;
  //   vec3 shadowCoord=vPositionFromLight.xyz/vPositionFromLight.w;
  // shadowCoord.xyz = (shadowCoord.xyz + 1.0) / 2.0;
  // float pcfBiasC = .2;
  // visibility = PCSS(uShadowMap, vec4(shadowCoord, 1.0), pcfBiasC);

  // vec3 phongColor=blinnPhong();
  
  // gl_FragColor=vec4(phongColor*visibility,1.);
    float visibility;
  float bias = 0.02;
   float filterRadiusUV = 3.0 / 2048.0;
  // vec3 shadowCoord=vPositionFromLight.xyz/1.0;
  // shadowCoord.xyz=(shadowCoord.xyz+1.0)/2.0;
  visibility = useShadowMap(uShadowMap, vec4(shadowCoord, 1.0), bias, 0.0);
  visibility = PCF(uShadowMap, vec4(shadowCoord, 1.0), bias, filterRadiusUV);
  visibility = PCSS(uShadowMap, vec4(shadowCoord, 1.0),bias);

  vec3 phongColor = blinnPhong();

  gl_FragColor = vec4(phongColor * visibility, 1.0);
  // gl_FragColor = vec4(phongColor, 1.0);
}