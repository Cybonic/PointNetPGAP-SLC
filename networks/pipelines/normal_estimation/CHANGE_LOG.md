08-04-2024: Changed the following to be compatible to the current pytorch version

In file pointnet2/_ex-src/include/utils.h  
// Define a macro to replace AT_CHECK with TORCH_CHECK
#define AT_CHECK TORCH_CHECK