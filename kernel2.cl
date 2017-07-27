f; // zn effect on zn deposition
    const float b   = 0.00001f; // zn effect on rb deposition
    const float dz  = 0.01569f;
    const float zncon = 0.01f;

    float synrate=(a*u+b*v*u+zncon)*exp(-1*alpha*v);//+diffu(ul, str);
    synrate=synu(synrate)-dz*u;
    return synrate;
}


inline float customgm(float u, float v, float alpha)
{
    // some constants

