f;

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











