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
    const float a   = 0.23787689208984375f; // zn effect on zn deposition
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

    const float c   = 2.06f;     // zn effect on rb deposition
    const float d   = 0.002f;    //4rb effect on rb deposition
    const float dr   = 0.1236f;     // rb diffusion rate
   	const float rbcon = 0.01f;
    
    float synrate=(c*v+d*v*u+rbcon)*exp(-1*alpha*v);
    synrate=synv(synrate)-dr*v;
    return synrate;
}
// =============================================================================


// bulk of work done here, evolves state
__kernel void timestep(
        int p_timestep,
        int p_filmstep,
        __global float * g_underlayer_in,
        __global float * g_uprev_in,
        __global float * g_vprev_in,
        __global float * g_unext_out,
        __global float * g_vnext_out,
        __global uint2 * g_keys_io,
        __global uint4 * g_ctrs_io)
{
    // get our unique identifiers
    int rowd = get_global_id(0);
    int row = (rowd + 1) % GRIDSIZE;
    int rowu = (row + 1) % GRIDSIZE;

    int cold = get_global_id(1);
    int col = (cold + 1) % GRIDSIZE;
    int colu = (col + 1) % GRIDSIZE;

    // this block will generate 4 random [0,1) numbers if uncommented,
    // accessed as randoms.x, randoms.y, randoms.z, randoms.w
    /*uint2 temp_key = (0u);
    uint4 temp_ctr = (0u);
    uint4 temp = (0u);
    float4 randoms = (0.0f);

    // copy from global to private
    temp_key = element(g_keys_io,row,col);
    temp_ctr = element(g_ctrs_io,row,col);

    // perform computation (macro, look in philox.cl if curious)
    next_philox(temp_ctr, temp_key, temp.x, temp.y, temp.z, temp.w)
    conv_randoms(temp_ctr, randoms)*/

    // some constants
    const float du = 0.02f;
    const float dv = 0.5f;
    const float r  = 1.0f;
    const float dt = 0.1f;
    const float dx = 0.6f;
    const float alpha = 10.0f;

    // the ul variable
    float ul = element(g_underlayer_in, row, col);

    // all the uprev elements: possible massive speedup with local memory
    float udd = element(g_uprev_in, rowd, cold);
    float ud0 = element(g_uprev_in, rowd,  col);
    float udu = element(g_uprev_in, rowd, colu);
    float u0d = element(g_uprev_in, row,  cold);
    float u00 = element(g_uprev_in, row,   col);
    float u0u = element(g_uprev_in, row,  colu);
    float uud = element(g_uprev_in, rowu, cold);
    float uu0 = element(g_uprev_in, rowu,  col);
    float uuu = element(g_uprev_in, rowu, colu);

    // all the vprev elements: local not implemented yet :(
    float vdd = element(g_vprev_in, rowd, cold);
    float vd0 = element(g_vprev_in, rowd,  col);
    float vdu = element(g_vprev_in, rowd, colu);
    float v0d = element(g_vprev_in, row,  cold);
    float v00 = element(g_vprev_in, row,   col);
    float v0u = element(g_vprev_in, row,  colu);
    float vud = element(g_vprev_in, rowu, cold);
    float vu0 = element(g_vprev_in, rowu,  col);
    float vuu = element(g_vprev_in, rowu, colu);
    
    // calculate unext
    float unext =
        u00+
        (r*customfm(u00,v00,alpha,p_filmstep,p_timestep)+
         du*(0.5f*(uuu+
                 uud+
                 udu+
                 udd)+
             (uu0+
              ud0+
              u0u+
              u0d)-
             6.0f*u00)/(dx*dx))*dt;
    unext = unext-(unext < 0.0f)*(unext-0.000001)-(unext > 20.0f)*(unext-20.0f);

    // calculate vnext
    float vnext = 
        v00+
        (r*customgm(u00,v00,alpha)+
         dv*(0.5f*(vuu+
                 vud+
                 vdu+
                 vdd)+
             (vu0+
              vd0+
              v0u+
              v0d)-
             6.0f*v00)/(dx*dx))*dt;
    vnext = vnext-(vnext < 0.0f)*(vnext-0.000001)-(vnext > 20.0f)*(vnext-20.0f);

    // write results to global
    element(g_unext_out,row,col) = unext;
    element(g_vnext_out,row,col) = vnext;
    
    // copy the cycled keys and ctrs back (otherwise we 
    // get the same set of randoms next pass!)
    /*element(g_keys_io,row,col) = temp_key;
    element(g_ctrs_io,row,col) = temp_ctr;*/
}

// merges u and v buffers into the underlayer buffer
__kernel void underlayer(
        __global float * g_uprev_in,
        __global float * g_vprev_in,
        __global float * g_underlayer_out,
        __global uint2 * g_keys_io,
        __global uint4 * g_ctrs_io)
{
    // get our unique identifiers
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    // this block will generate 4 random [0,1) numbers if uncommented,
    // accessed as randoms.x, randoms.y, randoms.z, randoms.w
    /*uint2 temp_key = (0u);
    uint4 temp_ctr = (0u);
    uint4 temp = (0u);
    float4 randoms = (0.0f);

    // copy from global to private
    temp_key = element(g_keys_io,row,col);
    temp_ctr = element(g_ctrs_io,row,col);

    // perform computation (macro, look in philox.cl if curious)
    next_philox(temp_ctr, temp_key, temp.x, temp.y, temp.z, temp.w)
    conv_randoms(temp_ctr, randoms)*/

    // copy to private memory
    float uprev = element(g_uprev_in,row,col);
    float vprev = element(g_vprev_in,row,col);

    // copy computation global memory
    element(g_underlayer_out,row,col) = (vprev > uprev)*1.0f;
    
    // copy the cycled keys and ctrs back (otherwise we 
    // get the same set of randoms next pass!)
    /*element(g_keys_io,row,col) = temp_key;
    element(g_ctrs_io,row,col) = temp_ctr;*/
}











