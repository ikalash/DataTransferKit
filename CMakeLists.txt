##---------------------------------------------------------------------------##
## DataTransferKit/CMakeLists.txt
## Stuart R. Slattery
##---------------------------------------------------------------------------##
## TriBITS package file for the DataTransferKit package
##---------------------------------------------------------------------------##

##---------------------------------------------------------------------------##
## Define the package
##---------------------------------------------------------------------------##

TRIBITS_PACKAGE_DECL(DataTransferKit)
TRIBITS_ADD_DEBUG_OPTION()
TRIBITS_ADD_SHOW_DEPRECATED_WARNINGS_OPTION()

##---------------------------------------------------------------------------##
## Set the path to the DTKData directory.
##---------------------------------------------------------------------------##

# Search for one of the mesh files in the DTKData directory
find_file(DTK_DATA_MESH_FILE cube_mesh.exo PATHS
    ${${PACKAGE_NAME}_SOURCE_DIR}/data/exodus
    ${${PACKAGE_NAME}_SOURCE_DIR}/DTKData/exodus
    NO_DEFAULT_PATH
)
if(DTK_DATA_MESH_FILE)
    # Get the location of the mesh file
    get_filename_component(DTK_DATA_DIR ${DTK_DATA_MESH_FILE} DIRECTORY)
    # Strip "/exodus" from the path
    string(REGEX REPLACE "/exodus" "" DTK_DATA_DIR ${DTK_DATA_DIR})
    message("-- DTKData found at ${DTK_DATA_DIR}")
else()
    set(DTK_DATA_DIR DTK_DATA_DIR-NOTFOUND)
    message("-- DTKData NOT found")
endif()

##---------------------------------------------------------------------------##
## Build the documentation published on http://datatransferkit.readthedocs.org
##---------------------------------------------------------------------------##

IF(DataTransferKit_ENABLE_ReadTheDocs)
    ADD_SUBDIRECTORY(docs)
ENDIF()

##---------------------------------------------------------------------------##
## Generate a .ycm_extra_conf.py file for use with YouCompleteMe
##---------------------------------------------------------------------------##

IF(DataTransferKit_ENABLE_YouCompleteMe)
    INCLUDE(CodeCompletion)
ENDIF()


##---------------------------------------------------------------------------##
## Do standard postprocessing
##---------------------------------------------------------------------------##

TRIBITS_PROCESS_SUBPACKAGES()
TRIBITS_PACKAGE_DEF()
TRIBITS_PACKAGE_POSTPROCESS()


##---------------------------------------------------------------------------##
## Set GlobalOrdinal type in DTK based on Tpetra 
##---------------------------------------------------------------------------##

SET(VALID_TPETRA_TYPE FALSE)
IF (Tpetra_INST_INT_LONG_LONG)
  TRIBITS_ADD_ETI_INSTANTIATIONS(TpetraCore
  "S={double} LO={int} GO={long long} N={Kokkos::Compat::KokkosSerialWrapperNode}"
  )
  SET(VALID_TPETRA_TYPE TRUE)
  message("-- Setting GO=long long in DTK")
ENDIF()
#IKT: the following is to allow other GO types besides long long.
#Note that to enable these, changes in packages/Interface/src/Client/DTK_Types.hpp
#are also required. 
#IF (Tpetra_INST_INT_LONG)
#  TRIBITS_ADD_ETI_INSTANTIATIONS(TpetraCore
#  "S={double} LO={int} GO={long} N={Kokkos::Compat::KokkosSerialWrapperNode}"
#  )
#  SET(VALID_TPETRA_TYPE TRUE)
#  message("-- Setting GO=long in DTK")
#ENDIF()
#IF (Tpetra_INST_INT_INT)
#  TRIBITS_ADD_ETI_INSTANTIATIONS(TpetraCore
#  "S={double} LO={int} GO={int} N={Kokkos::Compat::KokkosSerialWrapperNode}"
#  )
#  SET(VALID_TPETRA_TYPE TRUE)
#  message("-- Setting GO=int in DTK")
#ENDIF()
#IF (Tpetra_INST_INT_UNSIGNED) 
#  TRIBITS_ADD_ETI_INSTANTIATIONS(TpetraCore
#  "S={double} LO={int} GO={unsigned int} N={Kokkos::Compat::KokkosSerialWrapperNode}"
#  )
#  SET(VALID_TPETRA_TYPE TRUE)
#  message("-- Setting GO=unsigned int in DTK")
#ENDIF()
#IF(Tpetra_INST_INT_UNSIGNED_LONG) 
#  TRIBITS_ADD_ETI_INSTANTIATIONS(TpetraCore
#  "S={double} LO={int} GO={unsigned long} N={Kokkos::Compat::KokkosSerialWrapperNode}"
#  )
#  SET(VALID_TPETRA_TYPE TRUE)
#  message("-- Setting GO=unsigned long in DTK")
#ENDIF()

IF (NOT VALID_TPETRA_TYPE)
   message(FATAL_ERROR "DTK: invalid Global Ordinal type!  \n" 
           "Please recompile with -DTpetra_INST_INT_LONG_LONG:BOOL=ON\n")  
ENDIF()



##---------------------------------------------------------------------------##
## Print the revision number to stdout
##---------------------------------------------------------------------------##

FIND_PACKAGE(Git)
IF(GIT_FOUND)
    EXECUTE_PROCESS(
        COMMAND           ${GIT_EXECUTABLE} log --pretty=format:%H -n 1
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE   DTK_REVISION
        )
    MESSAGE("DTK Revision = '${DTK_REVISION}'")
ENDIF()

IF(Trilinos_ENABLE_DataTransferKitMoabAdapters)
    SET(MOAB_STAMP_FILE ${MOAB_LIBRARY_DIRS}/moab-revision)
    EXECUTE_PROCESS(COMMAND touch ${MOAB_STAMP_FILE})
    FILE(READ ${MOAB_STAMP_FILE} MOAB_REVISION)
    STRING(REGEX REPLACE "\n" "" MOAB_REVISION "${MOAB_REVISION}")
    MESSAGE("MOAB Revision = '${MOAB_REVISION}'")
ENDIF()

##---------------------------------------------------------------------------##
## Check C++ code style using clang-format version 3.9
##---------------------------------------------------------------------------##

IF(DataTransferKit_ENABLE_ClangFormat)
    INCLUDE(CodeFormat)
ENDIF()

##---------------------------------------------------------------------------##
## Check that there are no trailing whitespaces or tabs
##---------------------------------------------------------------------------##

FILE(WRITE
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/check_no_trailing_whitespaces_or_tabs.sh
    "#!/usr/bin/env bash\n"
    "\n"
    "grep "
    "--exclude={*.doc,*.eps,*.gif,*.jpg,*.pdf,*.png} "
    "--exclude={.project,.cproject} "
    "--exclude={*Makefile*,*makefile*} "
    "--exclude=.gitmodules "
    "--exclude-dir={.git,data} "
    "--regexp '[[:blank:]]$' "
    "--regexp $'\\t' "
    "--line-number "
    "`git ls-files`"
    "\n"
    "test \$? -eq 1"
)
FILE(COPY
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/check_no_trailing_whitespaces_or_tabs.sh
    DESTINATION
        ${CMAKE_BINARY_DIR}
    FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
)
ADD_TEST(
    NAME check_no_trailing_whitespaces_or_tabs
    COMMAND ${CMAKE_BINARY_DIR}/check_no_trailing_whitespaces_or_tabs.sh
    WORKING_DIRECTORY ${${PACKAGE_NAME}_SOURCE_DIR}
)

##---------------------------------------------------------------------------##
## end of DataTransferKit/CMakeLists.txt
##---------------------------------------------------------------------------##
