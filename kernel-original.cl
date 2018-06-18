/*
TODO : work_size : 3 ;

*/
__kernel void find_correspondences(__global int *intArgs, __global float *point_cloud_ptr, __global float *correspondence_result, __global float *input_transformed) {
  __private int i = get_global_id(0);
  __private int max_number_of_points = intArgs[3];

  if (i >= max_number_of_points) {
    return;
  }
  __private int point_cloud_ptr_size = intArgs[5];

  float a = 0.0;
  float b = 0.0;
  float c = 0.0;

  for (int k = 0; k< point_cloud_ptr_size; k++  ) {
    a = (input_transformed[3*i] - point_cloud_ptr[3*k])*(input_transformed[3*i] - point_cloud_ptr[3*k]);
    b = (input_transformed[3*i+1] - point_cloud_ptr[3*k+1])*(input_transformed[3*i+1] - point_cloud_ptr[3*k+1]);
    c = (input_transformed[3*i+2] - point_cloud_ptr[3*k+2])*(input_transformed[3*i+2] - point_cloud_ptr[3*k+2]);
    //if (dis<=0.5) {
    if (sqrt(a+b+c)<0.02f) {
      correspondence_result[3*i]= (float)i;
      correspondence_result[3*i+1] =(float)k;
      correspondence_result[3*i+2] = sqrt(a+b+c);
      k = point_cloud_ptr_size;
    }
  }
  //Subject to Change
  //correspondence_result_count[i]=a+b+c;
}


// TUAN : Line 374 global_classification
/*
  1. floatArgs : Collections of arguments with float data type like : float angle_min, float angle_max, float angle_step, float shift_min, float shift_max, float shift_step,
  2. initialTranslation :
  3. direction:
  4. model_voxelized: to be shifted PointCloud (correspondence)
  5. point_cloud_ptr: original PointCloud (correspondence)
  6. rotation
  7. correspondence_result: result of all between point clouds
  8. correspondence_count : count of correspondences
  9. work_size_dimension :
  10. model_voxelized_size:
  11. point_cloud_ptr_size:
*/
__kernel void shiftAndRollWithoutSumLoop(__global float *floatArgs, __global float *initialTranslation, __global float *direction,__global float *model_voxelized, __global float *point_cloud_ptr, __global float *rotation, __global float *correspondence_result,__global int *correspondence_result_count, __global int *work_size_dimension, __global int *sources_size, __global float *input_transformed) {
}

/*
  floatArgs:
  initialTranslation : 6-8
  direction :  9-11
  rotation :12-20

  TODO :
  work_size_dimension = 0-2
*/
__kernel void transforming_models(__global float *floatArgs,__global float *model_voxelized, __global int *work_size_dimension,  __global float *input_transformed) {
  __private int angle = get_global_id(0);
  __private int shift = get_global_id(1);
  __private int point = get_global_id(2);

  __private float angle_min = floatArgs[0];
  __private float angle_step = floatArgs[2];
  __private float shift_min  = floatArgs[3];
  __private float shift_step = floatArgs[5];

  __private int number_angle_step = work_size_dimension[0];
  __private int number_shift_step = work_size_dimension[1];
  __private int model_voxelized_size = work_size_dimension[2];

  __private float rotating[9]= {};
  __private float transform[16]= {};
  __private int start_index = (number_shift_step*angle+shift)*model_voxelized_size*3;
  __private float angle_temp =(angle_min+angle*angle_step)*(0.01745328888);

  rotating[0] = cos(angle_temp);
  rotating[1] = -sin(angle_temp);
  rotating[3] = sin(angle_temp);
  rotating[4] = cos(angle_temp);

  __private float shift_temp = shift_min + shift*shift_step;

  transform[0]= floatArgs[12]*rotating[0]+floatArgs[13]*rotating[3];
  transform[1]= floatArgs[12]*rotating[1]+floatArgs[13]*rotating[4];
  transform[2]= floatArgs[14];

  transform[4]= floatArgs[15]*rotating[0]+floatArgs[16]*rotating[3];
  transform[5]= floatArgs[15]*rotating[1]+floatArgs[16]*rotating[4];
  transform[6]= floatArgs[17];

  transform[8]= floatArgs[18]*rotating[0]+floatArgs[19]*rotating[3];
  transform[9]= floatArgs[18]*rotating[1]+floatArgs[19]*rotating[4];
  transform[10]=floatArgs[20];


  transform[3] = floatArgs[6]+ floatArgs[9]*shift_temp/floatArgs[11];
  transform[7] =floatArgs[7]+ floatArgs[10]*shift_temp/floatArgs[11];
  transform[11] =floatArgs[8]+ floatArgs[11]*shift_temp/floatArgs[11];

  __private bool ident = true;

  for (int i = 0 ; i < 4 ; i++) {
    for (int k = 0; k < 4 ; k++) {
      if (i == k ) {
        if (transform[i*4+k]!= 1.0f) {
          ident = false;
          break;
        }
      }
      else {
        if(transform[i*4+k]!= 0.0f) {
          ident = false;
          break;
        }
      }
    }
  }

  int i = point;
  if (!ident) {
    input_transformed[start_index + 3*i] = model_voxelized[3*i]*transform[0] + model_voxelized[3*i+1]*transform[1] + model_voxelized[ 3*i+2]*transform[2]+transform[3];
    input_transformed[start_index + 3*i+1] = model_voxelized[3*i]*transform[4] + model_voxelized[3*i+1]*transform[5] + model_voxelized[3*i+2]*transform[6]+transform[7];
    input_transformed[start_index + 3*i+2] = model_voxelized[3*i]*transform[8] + model_voxelized[3*i+1]*transform[9] + model_voxelized[3*i+2]*transform[10]+transform[11];
  }
  else {
    input_transformed[start_index+3*i]=model_voxelized[3*i];
    input_transformed[start_index+3*i+1]=model_voxelized[3*i+1];
    input_transformed[start_index+3*i+2]=model_voxelized[3*i+2];
  }

}

__kernel void computeDifferencesForCorrespondence(__global float *correspondence_result, __global int *work_sizes,  __global int *correspondence_result_count) {
    //angle
    int i  = get_global_id(0);
    //shift
    int k  = get_global_id(1);

    int num_angle_steps = work_sizes[0];
    int num_shift_steps = work_sizes[1];

    __private int model_voxelized_size = work_sizes[2];

    int start_index = (num_shift_steps*i+k)*model_voxelized_size;
    int count = 0;
    for (int i = 0 ; i<model_voxelized_size; i++ ){
      if (correspondence_result[3*(i+start_index)+2]!= 0) {
        count++;
      }
    }
    correspondence_result_count[(num_shift_steps*i+k)*3] = i;
    correspondence_result_count[(num_shift_steps*i+k)*3+1] =k;
    correspondence_result_count[(num_shift_steps*i+k)*3+2] = count;
    //4.5 : TODO : from here

}

//https://stackoverflow.com/questions/7627098/what-is-a-lambda-expression-in-c11
/*
  Files to do this :
  1. https://github.com/PointCloudLibrary/pcl/blob/master/registration/include/pcl/registration/impl/correspondence_estimation.hpp#L113
  2. https://github.com/PointCloudLibrary/pcl/blob/master/registration/include/pcl/registration/correspondence_estimation.h#L63
  3. https://github.com/PointCloudLibrary/pcl/blob/master/common/include/pcl/pcl_base.h
*/
