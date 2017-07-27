f;     // zn effect on rb deposition
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
