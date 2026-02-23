# ------------------------------------------------------------------------------
# List of executables to test the compressible module
# ------------------------------------------------------------------------------

set(TEST_COMPRESSIBLE ${TEST_COMPRESSIBLE}
   Test_compressible
   Uniform_flow_compressible
   Uniform_flow_compressible_ChProblem
   Test_shock_tube_3D
   Test_shock_tube_3D_variable_mass
   Test_shock_tube_1D
   Test_shock_tube_1D_v2
   Test_shock_tube_1D_smoothed
   Test_shock_tube_2D
   Test_gas_expansion
   Test_blast_wave_1D
   Test_blast_wave_1D_smoothed
   Test_blast_wave_2D

)

#--------------------------------------------------------------

list(APPEND LIBS Chrono_core)
list(APPEND LIBS "Chrono_fsisph")
list(APPEND LIBS "Chrono_fsisph_compressible")

if(CH_ENABLE_MODULE_VSG)
    list(APPEND LIBS "Chrono_fsisph_vsg")
    list(APPEND LIBS "Chrono_fsisph_compressible_vsg")
endif()

if(CH_ENABLE_MODULE_PARDISO_MKL)
    list(APPEND LIBS "Chrono_pardisomkl")
endif()

#--------------------------------------------------------------

message(STATUS "Add test programs for FSI COMPRESSIBLE module")



foreach(PROGRAM ${TEST_COMPRESSIBLE})
    
    add_executable(${PROGRAM} ${PROGRAM}.cpp)
    source_group("" FILES ${PROGRAM}.cpp)

    if(APPLE)
      set_target_properties(${PROGRAM} PROPERTIES MACOSX_BUNDLE_INFO_PLIST ${MACOSX_BUNDLE_INFO_PLIST})
    endif()

    if(MSVC)
      set_target_properties(${PROGRAM} PROPERTIES MSVC_RUNTIME_LIBRARY ${CH_MSVC_RUNTIME_LIBRARY})
    endif()
    
    set_property(TARGET ${PROGRAM} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:${PROGRAM}>")
 
    target_link_libraries(${PROGRAM} ${LIBS})

    install(TARGETS ${PROGRAM} CONFIGURATIONS Release DESTINATION ${CH_INSTALL_DEMO})
endforeach()


#------------------------------------------------------------------
# Add a test executable to verify the Custom Compressible module
#------------------------------------------------------------------
#add_executable(Test_compressibility Test_compressible.cpp)
#source_group("" FILES Test_compressible.cpp)
#if(APPLE)
#   set_target_properties(Test_compressibility PROPERTIES MACOSX_BUNDLE_INFO_PLIST ${MACOSX_BUNDLE_INFO_PLIST})
#endif()
#if(MSVC)
#   set_target_properties(Test_compressibility PROPERTIES MSVC_RUNTIME_LIBRARY ${CH_MSVC_RUNTIME_LIBRARY})
#endif()
    
#set_property(TARGET Test_compressibility PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:${PROGRAM}>")
#list(APPEND LIBS "Chrono_fsisph_compressible")
#target_link_libraries(Test_compressibility ${LIBS} )

#install(TARGETS ${PROGRAM} CONFIGURATIONS Release DESTINATION ${CH_INSTALL_DEMO})

