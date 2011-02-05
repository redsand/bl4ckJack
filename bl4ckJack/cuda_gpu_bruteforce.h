


/* 
    Copyright (C) 2009  Benjamin Vernoux, titanmkd@gmail.com

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 3 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA */

#ifndef CUDA_BRUTEFORCE_GPU_H
#define CUDA_BRUTEFORCE_GPU_H

#ifdef  __cplusplus
extern "C" {
#endif

	extern int GPUBruteforceInit(char *charset, char charset_len);

	extern int GPUBruteforceKernelExecute(int blocks_x, int blocks_y, int threads_per_block, int shared_mem_required, char *src_gwords, char *dst_gwords);

#ifdef  __cplusplus
}
#endif

#endif /* CUDA_BRUTEFORCE_GPU_H */

