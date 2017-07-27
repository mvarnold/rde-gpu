// this is my prng gpu code, based on a 7 round philox algorithm
#include "philox.cl"

/* Definitions provided by host code
 * GRIDSIZE - number of elements along one side of our box
 * TIMESTEPS - total number of timesteps to perform
 * FILMSTEPS - total number of filmsteps to perform
 */

// convenient define
#define element(buff,row,col) buff[row*GRIDSIZE + col]

// this function initializes the u and v buffers 
__kernel void initialize(
        __global uint2 * g_keys_io,
        __global uint4 * g_ctrs_io,
        __global float * g_uprev_out,
        __global float * g_vprev_out)
{
    // get our unique identifiers
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    // private memory
    uint2 temp_key = (0u);
    uint4 temp_ctr = (0u);
    uint4 temp = (0u);
    float4 randoms = (0.0f);

    // copy from global to private
    temp_key = element(g_keys_io,row,col);
    temp_ctr = element(g_ctrs_io,row,col);

    // perform computation (macro, look in philox.cl if curious)
    next_philox(temp_ctr, temp_key, temp.x, temp.y, temp.z, temp.w)
    conv_randoms(temp_ctr, randoms)

    // write from private to global
    element(g_uprev_out,row,col) = 0.01f*randoms.x + 0.1f;
    element(g_vprev_out,row,col) = 0.01f*randoms.y + 0.1f;
    element(g_keys_io,row,col) = temp_key;
    element(g_ctrs_io,row,col) = temp_ctr;
}

// convenience functions for timestep
inline float synu(float synrate)
{
    return synrate - (synrate < 0.0f)*synrate - (synrate > 1.0f)*(synrate - 1.0f);
}

inline float synv(float synrate)
{
    return synrate - (synrate < 0.0f)*synrate - (synrate > 1.0f)*(synrate - 1.0f);
}

inline float diffu(float ul,float strength)
{
    return 0.01f*(ul!=1.0f)*strength;
}

inline float diffv(float ul,float strength)
{
    return 0.1f*(ul==1.0f)*strength;
}

inline float concentration(int filmstep, int timestep)
{
    //float c =  0.005f + 0.033f *(float)(filmstep)/(float)(FILMSTEPS/2);
    float c=0.1f;
    //if (filmstep > FILMSTEPS/2)
    //    return 0.03f;
    //else
        return c;
}

inline float linearf(float u, float v, float ul, int filmstep, int timestep)
{
    // some constants
    const float str = 0.0008f;

    const float a   = 0.08f;
    const float b   = -0.08f;
    const float c   = 0.018;    

    float d = 0.02f;
    float synrate=a*u+b*v+c;//+diffu(ul, str);
    synrate=synu(synrate);
    synrate=synrate-d*u;
    return synrate;
}


inline float linearg(float u, float v, float ul)
{
    // some constants
    const float str = 0.0008f;
    const float e   = 0.1f;
    const float f   = 0.15f;

    float g=0.08f;
    float synrate=e*u-f; //+diffv(ul,str);
    synrate=synv(synrate);
    synrate=synrate-g*v;
    return synrate;
}

inline float customf(float u, float v, float alpha, int filmstep, int timestep)
{
    // some constants
    const float a   = 0.0001f;
    const float b   = 0.01f;
    const float c   = 0.2f;     ;  //concentration(filmstep, timestep);
    
    float synrate=a*u+c*exp(-1*alpha*u/v);//+diffu(ul, str);
    synrate=synu(synrate)-b*u;
    return synrate;
}


inline float customg(float u, float v, float alpha)
{
    // some constants
    
    const float e   = 0.0001f;
    const float f   = 0.05f;
    const float g   = 0.8f;
    const float h   = 1.0f;
    
    float synrate=e*v+(g+h*u)*exp(-1*alpha*u/v);
    synrate=synv(synrate)-f*v;
    return synrate;
}
// =============================================================================
inline float customfm(float u, float v, float alpha, int filmstep, int timestep)
{
    // some constants
