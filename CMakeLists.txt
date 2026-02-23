#=============================================================================
# CMake configuration file for Chrono FSI module
#
# Cannot be used stand-alone (it is loaded by parent CMake configuration file)
#=============================================================================

option(CH_ENABLE_MODULE_FSI "Enable the Chrono FSI module" OFF)    # sets a cached variable the first time it is run

if(NOT CH_ENABLE_MODULE_FSI)
  mark_as_advanced(FORCE CH_USE_FSI_DOUBLE)
  return()
else()
  mark_as_advanced(CLEAR CH_USE_FSI_DOUBLE)
endif()

message(STATUS "\n==== Chrono FSI module ====\n")

#-------------------------------------------------------------------------------
# Create the generic FSI interface library
#-------------------------------------------------------------------------------

message(STATUS "Generic Chrono::FSI interface library")

set(FSI_FILES
    ChApiFsi.h
    ChFsiDefinitions.h
    ChFsiSystem.h
    ChFsiSystem.cpp
    ChFsiFluidSystem.h
    ChFsiFluidSystem.cpp
    ChFsiInterface.h
    ChFsiInterface.cpp
    ChFsiBenchmark.h
)
source_group("" FILES ${FSI_FILES})           # no influence on build and compilation. Just organizes how source files are showed inside an IDE. With "" placed in root dir

message(STATUS "  Add Chrono_fsi library")

add_library(Chrono_fsi
    ${FSI_FILES}
    ${FSI_UTILS_FILES}
)                                             # Chrono_fsi to be built as library. Type omitted, so defaults to value of BUILD_SHARED_LIBS variable
                                              # Chrono_fsi includes only standard fsi source files. SPH specific files grouped in the Chrono_fsisph library in sph subdir.
                                              #FSI_UTILS_FILES not defined in parent scope. Maybe defined in subdir and propagated. If not defined not an error, expands to empty

add_library(Chrono::fsi ALIAS Chrono_fsi)                          

set_target_properties(Chrono_fsi PROPERTIES DEBUG_POSTFIX ${CH_DEBUG_POSTFIX})      # tells cmake that Chrono_fsi lib in debug mode needs postfix set by variable (= _d)

if (CH_STATIC)        # variable refers to build shared or static libs
  set_target_properties(Chrono_fsi PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(MSVC)
  target_compile_options(Chrono_fsi PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/wd4251>)  # missing DLL interface
  set_target_properties(Chrono_fsi PROPERTIES MSVC_RUNTIME_LIBRARY ${CH_MSVC_RUNTIME_LIBRARY})
endif()

target_link_libraries(Chrono_fsi PRIVATE Chrono_core)         # Chrono_fsi lib depends on Chrono_core lib.

target_compile_definitions(Chrono_fsi PRIVATE $<$<COMPILE_LANGUAGE:CXX>:CH_API_COMPILE_FSI>)       # at compile time, for language C++ only, pass the option CH_API_COMPILE_FSI
                                                                                                   # Equivalent to pass -DCH_API_COMPILE_FSI to compiler. It defines the 
                                                                                                   # CH_API_COMPILE_FSI macro, so availble during program execution.
                                                                                                   # Defines without assigning it a value. Macro useful in ChApiFsi.h file to mark 
                                                                                                   # the module as imported or as exported.
                                                                                                   

install(TARGETS Chrono_fsi
        EXPORT ChronoTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        INCLUDES DESTINATION include/chrono_fsi)              # Install command to specify targets and locations of headers, when calling"cmake --install" (or INSTALL target)
                                                              # Here with the EXPORT keyword we are just telling CMake to mark the target Chrono_fsi as part of the Chrono_targets
                                                              # export set, we are not actually creating the file ChronoTargets.cmake
                                                              # In chrono, all installed targets will be marked as part of the same export set. In this way, at the end step we
                                                              # call install(EXPORT ChronoTargets) from the top level dir, and in the install directory the generated 
                                                              # ChronoTargets.cmake will include all parts called from various "submodules", and to import it we only need this   
                                                              # single file to receive all targets. 

#-------------------------------------------------------------------------------
# Configure submodules
#-------------------------------------------------------------------------------
add_subdirectory(sph)

#---------------------------------------------------------------
# Add custom submodule
#---------------------------------------------------------------

add_subdirectory(sph_compressible)

# Custom header files should be installed by default by following install command still preserving directory structure


#-------------------------------------------------------------------------------
# Install header files of this folder and subfolders in the include directory of install destination
#-------------------------------------------------------------------------------

# Old way (install all Chrono::FSI headers .h and .cuh into include/chrono_fsi destination folder preserving directory structure)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
DESTINATION include/chrono_fsi
FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh")

# Install 3rd party headers       
install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/chrono_thirdparty/stb
        DESTINATION include/chrono_thirdparty
        FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh" PATTERN "*.hpp" PATTERN "*.inl")
