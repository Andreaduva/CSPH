#--------------------------------------------------------------------------------
# CMakeLists file of folder chrono_fsi/sph_compressible
# No need to ask user's permission. Use options already given for FSI::SPH module
#--------------------------------------------------------------------------------

message(STATUS "Custom compressible module for FSI::SPH")
# Ensure the FSI:SPH module has been built successfully
if(NOT CH_ENABLE_MODULE_FSI)
   message(STATUS "Chrono::FSI::SPH module not enabled, custom module can't be built")
   return()
endif()

#-----------------------------------------------------------------------------
# List all files for the SPH-based Chrono::FSI library
# Associate source files to string variables, configure source groups for convenient 
# display in IDEs
#-----------------------------------------------------------------------------

set(FSISPH_COMPRESSIBLE_BASE_FILES 
     ChFsiDefinitionsSPH_compressible.h
     ChFsiParamsSPH_compressible.h
     ChFsiFluidSystemSPH_compressible.h
     ChFsiFluidSystemSPH_compressible.cpp
     ChFsiInterfaceSPH_compressible.h
     ChFsiInterfaceSPH_compressible.cpp
     ChFsiSystemSPH_compressible.h
     ChFsiSystemSPH_compressible.cpp
     ChFsiProblemSPH_compressible.h
     ChFsiProblemSPH_compressible.cpp)

source_group("" FILES ${FSISPH_COMPRESSIBLE_BASE_FILES})


set(FSISPH_COMPRESSIBLE_UTILS_FILES
    utils/UtilsPrintSph_compressible.cuh
    utils/UtilsPrintSph_compressible.cu
    utils/UtilsDevice_compressible.cuh
)
source_group("utils" FILES ${FSISPH_COMPRESSIBLE_UTILS_FILES})


set(FSISPH_COMPRESSIBLE_PHYSICS_FILES
    physics/FsiDataManager_compressible.cuh
    physics/FsiDataManager_compressible.cu
    physics/SphGeneral_compressible.cuh
    physics/SphGeneral_compressible.cu
    physics/CollisionSystem_compressible.cuh
    physics/CollisionSystem_compressible.cu
    physics/FsiParticleRelocator_compressible.cuh
    physics/FsiParticleRelocator_compressible.cu
    physics/BceManager_compressible.cuh
    physics/BceManager_compressible.cu
    physics/FsiForce_base.cuh
    physics/FsiForce_base.cu
    physics/FsiForce_compressible.cuh
    physics/FsiForce_compressible.cu
    physics/FluidDynamics_compressible.cuh
    physics/FluidDynamics_compressible.cu
    )
source_group("physics" FILES ${FSISPH_COMPRESSIBLE_PHYSICS_FILES})

set(FSISPH_COMPRESSIBLE_MATH_FILES
    math/CustomMath_compressible.cuh
    )
source_group("math" FILES ${FSISPH_COMPRESSIBLE_MATH_FILES})

# DEBUG COMMAND NOT WORKING
# set_source_files_properties(physics/FsiDataManager_compressible.cu PROPERTIES LANGUAGE CUDA)

message(STATUS "FSISPH_COMPRESSIBLE_BASE_FILES files are ${FSISPH_COMPRESSIBLE_BASE_FILES}")
message(STATUS "FSISPH_COMPRESSIBLE_UTILS_FILES files are ${FSISPH_COMPRESSIBLE_UTILS_FILES}")
message(STATUS "FSISPH_COMPRESSIBLE_PHYSICS_FILES files are ${FSISPH_COMPRESSIBLE_PHYSICS_FILES}")
message(STATUS "FSISPH_COMPRESSIBLE_MATH_FILES files are ${FSISPH_COMPRESSIBLE_MATH_FILES}")

# --------------- 3rd party STB FILES------------------------------------------

# ---- Compilation of following files could be avoided since already in original sph lib, but if not done 
# ---- generation of weird linker errors, so keep the duplicate

set(FSISPH_COMPRESSIBLE_STB_FILES
   ${CMAKE_SOURCE_DIR}/src/chrono_thirdparty/stb/stb.h
   ${CMAKE_SOURCE_DIR}/src/chrono_thirdparty/stb/stb_image.h
   ${CMAKE_SOURCE_DIR}/src/chrono_thirdparty/stb/stb_image.cpp
   ${CMAKE_SOURCE_DIR}/src/chrono_thirdparty/stb/stb_image_write.h
   ${CMAKE_SOURCE_DIR}/src/chrono_thirdparty/stb/stb_image_write.cpp
 )
source_group("stb" FILES ${FSISPH_COMPRESSIBLE_STB_FILES})

# --------------- COMPRESSIBLE SPH FSI VSG INTERFACE----------------------------------------
# VSG visualization for fsi enabled only if main VSG module installed as well
# -----------------------------------------------------------------------------

if(CH_ENABLE_MODULE_VSG)
    set(FSISPH_COMPRESSIBLE_VSG_FILES
        visualization/ChFsiVisualizationVSG_compressible.h
        visualization/ChFsiVisualizationVSG_compressible.cpp
    )
else()
    set(FSISPH_COMPRESSIBLE_VSG_FILES "")
endif()
source_group("" FILES ${FSISPH_COMPRESSIBLE_VSG_FILES})



# ------------------------------------------------------------------------------
# Set dependencies on other Chrono modules, here VSG, and propagate to parent scope
# ------------------------------------------------------------------------------

set(DEPENDENCIES_FSI "")
if(CH_ENABLE_MODULE_VSG)
  list (APPEND DEPENDENCIES_FSI "VSG")
endif()

set(DEPENDENCIES_FSI ${DEPENDENCIES_FSI} PARENT_SCOPE)       

#-----------------------------------------------------------------------------
# Create the Custom Chrono_fsisph_compressible library
#-----------------------------------------------------------------------------

message(STATUS "  Add Chrono_fsisph_compressible library")

add_library(Chrono_fsisph_compressible
    ${FSISPH_COMPRESSIBLE_BASE_FILES}
    ${FSISPH_COMPRESSIBLE_PHYSICS_FILES}
    ${FSISPH_COMPRESSIBLE_MATH_FILES}
    ${FSISPH_COMPRESSIBLE_UTILS_FILES}
    ${FSISPH_COMPRESSIBLE_STB_FILES}
)
add_library(Chrono::fsisph_compressible ALIAS Chrono_fsisph_compressible)

set_target_properties(Chrono_fsisph_compressible PROPERTIES DEBUG_POSTFIX ${CH_DEBUG_POSTFIX})

if (CH_STATIC)
  set_target_properties(Chrono_fsisph_compressible PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(MSVC)
  target_compile_options(Chrono_fsisph_compressible PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/wd4251>)              # missing DLL interface
  target_compile_options(Chrono_fsisph_compressible PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/wd4100>)              # unreferenced formal parameter
  target_compile_options(Chrono_fsisph_compressible PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4324>)  # unreferenced formal parameter
  set_target_properties(Chrono_fsisph_compressible PROPERTIES MSVC_RUNTIME_LIBRARY ${CH_MSVC_RUNTIME_LIBRARY})
endif()

target_link_libraries(Chrono_fsisph_compressible PRIVATE Chrono_core)
#target_link_libraries(Chrono_fsisph_compressible PUBLIC Chrono_fsi)
target_link_libraries(Chrono_fsisph_compressible PUBLIC Chrono_fsisph)

target_link_libraries(Chrono_fsisph_compressible PUBLIC CUDA::cudart_static)
target_link_libraries(Chrono_fsisph_compressible PUBLIC CUDA::nvrtc)
target_link_libraries(Chrono_fsisph_compressible PUBLIC CUDA::cuda_driver)
target_link_libraries(Chrono_fsisph_compressible PUBLIC CUDA::cublas)
target_link_libraries(Chrono_fsisph_compressible PUBLIC CUDA::cusparse)
target_link_libraries(Chrono_fsisph_compressible PUBLIC Thrust::Thrust)

target_compile_definitions(Chrono_fsisph_compressible PRIVATE "THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA")
target_compile_definitions(Chrono_fsisph_compressible PRIVATE "THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP")

target_compile_definitions(Chrono_fsisph_compressible PRIVATE $<$<COMPILE_LANGUAGE:CXX>:CH_API_COMPILE_FSI>)

set_target_properties(Chrono_fsisph_compressible PROPERTIES CUDA_ARCHITECTURES ${CHRONO_CUDA_ARCHITECTURES})

#DEBUG COMMAND FOR THE LINKING ERROR NOT WORKING
# set_property(TARGET Chrono_fsisph_compressible PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS Chrono_fsisph_compressible
        EXPORT ChronoTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        INCLUDES DESTINATION include/chrono_fsi) 



#-------------------------------------------------------------------------------
# Add the Chrono_fsisph_compressible_vsg library, if the main VSG Chrono module enabled.
#-------------------------------------------------------------------------------

if(CH_ENABLE_MODULE_VSG)
    message(STATUS "  Add Chrono_fsisph_compressible_vsg library")

    add_library(Chrono_fsisph_compressible_vsg
        ${FSISPH_COMPRESSIBLE_VSG_FILES}
    )
    add_library(Chrono::fsisph_compressible_vsg ALIAS Chrono_fsisph_compressible_vsg)

    set_target_properties(Chrono_fsisph_compressible_vsg PROPERTIES DEBUG_POSTFIX ${CH_DEBUG_POSTFIX})

   
    if (CH_STATIC)
      set_target_properties(Chrono_fsisph_compressible_vsg PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
 
    if(MSVC)
      # disable warnings related to missing DLL interface
      target_compile_options(Chrono_fsisph_compressible_vsg PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/wd4251>)
      set_target_properties(Chrono_fsisph_compressible_vsg PROPERTIES MSVC_RUNTIME_LIBRARY ${CH_MSVC_RUNTIME_LIBRARY})
    endif()

    target_compile_definitions(Chrono_fsisph_compressible_vsg PRIVATE $<$<COMPILE_LANGUAGE:CXX>:CH_API_COMPILE_FSI>)
    
    target_link_libraries(Chrono_fsisph_compressible_vsg PRIVATE Chrono_core Chrono_fsisph Chrono_fsisph_compressible)
    target_link_libraries(Chrono_fsisph_compressible_vsg PUBLIC Chrono_vsg)

    target_link_libraries(Chrono_fsisph_compressible_vsg PRIVATE CUDA::cudart_static)
    target_link_libraries(Chrono_fsisph_compressible_vsg PRIVATE CUDA::nvrtc)
    target_link_libraries(Chrono_fsisph_compressible_vsg PRIVATE CUDA::cuda_driver)
    target_link_libraries(Chrono_fsisph_compressible_vsg PRIVATE CUDA::cublas)
    target_link_libraries(Chrono_fsisph_compressible_vsg PRIVATE CUDA::cusparse)

    target_link_libraries(Chrono_fsisph_compressible_vsg PRIVATE Thrust::Thrust)
    target_compile_definitions(Chrono_fsisph_compressible_vsg PRIVATE "THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA")
    target_compile_definitions(Chrono_fsisph_compressible_vsg PRIVATE "THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP")

    install(TARGETS Chrono_fsisph_compressible_vsg
            EXPORT ChronoTargets
            RUNTIME DESTINATION bin
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            INCLUDES DESTINATION include/chrono_fsi)      # installed and marked as exported target in ChronoTargets as all other targets.

endif()

