##---------------------------------------------------------------------------##
## DataTransferKit/CMakeLists.txt
## Stuart R. Slattery
##---------------------------------------------------------------------------##
## TriBITS package file for the DataTransferKit package
##---------------------------------------------------------------------------##

##---------------------------------------------------------------------------##
## Define the package
##---------------------------------------------------------------------------##

TRIBITS_PACKAGE(DataTransferKit)
TRIBITS_ADD_DEBUG_OPTION()
TRIBITS_ADD_SHOW_DEPRECATED_WARNINGS_OPTION()

##---------------------------------------------------------------------------##
## Explicit template instantiation
##---------------------------------------------------------------------------##

# Setup the same ETI as Tpetra but do not set global ordinal types as
# additional scalar types.

# check tpetra variables are defined.
ASSERT_DEFINED(TpetraCore_ETI_SCALARS)
ASSERT_DEFINED(TpetraCore_ETI_LORDS)
ASSERT_DEFINED(TpetraCore_ETI_GORDS)
ASSERT_DEFINED(TpetraCore_ETI_NODES)

# set the eti types
SET (${PACKAGE_NAME}_ETI_SCALARS_TEMP ) # start with the empty list
FOREACH (scalarName IN LISTS TpetraCore_ETI_SCALARS)
  LIST (FIND TpetraCore_ETI_GORDS ${scalarName} scalarNameIndex)
  IF (-1 EQUAL scalarNameIndex) # not in the list of integer types
    LIST (APPEND ${PACKAGE_NAME}_ETI_SCALARS_TEMP ${scalarName})
  ENDIF ()
ENDFOREACH ()

SET(${PACKAGE_NAME}_ETI_SCALARS ${${PACKAGE_NAME}_ETI_SCALARS_TEMP})
SET(${PACKAGE_NAME}_ETI_LORDS   ${TpetraCore_ETI_LORDS})
SET(${PACKAGE_NAME}_ETI_GORDS   ${TpetraCore_ETI_GORDS})
SET(${PACKAGE_NAME}_ETI_NODES   ${TpetraCore_ETI_NODES})

# make the eti variables globally accessible
GLOBAL_SET(${PACKAGE_NAME}_ETI_SCALARS ${${PACKAGE_NAME}_ETI_SCALARS})
GLOBAL_SET(${PACKAGE_NAME}_ETI_LORDS   ${${PACKAGE_NAME}_ETI_LORDS})
GLOBAL_SET(${PACKAGE_NAME}_ETI_GORDS   ${${PACKAGE_NAME}_ETI_GORDS})
GLOBAL_SET(${PACKAGE_NAME}_ETI_NODES   ${${PACKAGE_NAME}_ETI_NODES})

##---------------------------------------------------------------------------##
## ETI
##---------------------------------------------------------------------------##

INCLUDE(TribitsETISupport)

TRIBITS_ADD_EXPLICIT_INSTANTIATION_OPTION()

TRIBITS_ADD_ETI_SUPPORT()

ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_EXPLICIT_INSTANTIATION)

TRIBITS_ADD_OPTION_AND_DEFINE(
    ${PACKAGE_NAME}_ENABLE_EXPLICIT_INSTANTIATION
    HAVE_DATATRANSFERKIT_EXPLICIT_INSTANTIATION
    "Enable explicit template instantiation (ETI) in DTK"
    ${${PACKAGE_NAME}_ENABLE_EXPLICIT_INSTANTIATION}
    )

INCLUDE("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ETI.cmake")

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
## Print the revision number to stdout
##---------------------------------------------------------------------------##

FIND_PACKAGE(Git)
IF(GIT_FOUND AND EXISTS ${${PACKAGE_NAME}_SOURCE_DIR}/.git)
    EXECUTE_PROCESS(
        COMMAND           ${GIT_EXECUTABLE} log --pretty=format:%H -n 1
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE   ${PACKAGE_NAME}_GIT_COMMIT_HASH
        )
ELSE()
    SET(${PACKAGE_NAME}_GIT_COMMIT_HASH "Not a git repository")
ENDIF()
MESSAGE("DTK Revision = '${${PACKAGE_NAME}_GIT_COMMIT_HASH}'")


# Design-by-Contract
TRIBITS_ADD_OPTION_AND_DEFINE(
    ${PACKAGE_NAME}_ENABLE_DBC
    HAVE_DTK_DBC
    "Enable Design-by-Contract checks. WARNING: use for debug checking but disable for production runs as it incurs a significant overhead."
    ${${PROJECT_NAME}_ENABLE_DEBUG}
)

##---------------------------------------------------------------------------##
## Set extra parameters (before calling CONFIGURE_FILE)
##---------------------------------------------------------------------------##

IF( HAVE_DATATRANSFERKIT_BOOSTORG )
  GLOBAL_SET( HAVE_DTK_BOOST TRUE )
ENDIF()

# Check for netcdf so we can use it in the tests.
IF( HAVE_DATATRANSFERKIT_NETCDF )
  GLOBAL_SET( HAVE_DTK_NETCDF TRUE )
ENDIF()


##---------------------------------------------------------------------------##
## Do standard postprocessing
##---------------------------------------------------------------------------##

INCLUDE(${PACKAGE_SOURCE_DIR}/Version.cmake)
TRIBITS_CONFIGURE_FILE(${PACKAGE_NAME}_config.hpp)

ADD_SUBDIRECTORY(packages)

TRIBITS_PACKAGE_POSTPROCESS()

##---------------------------------------------------------------------------##
## Check C++ code style using clang-format
##---------------------------------------------------------------------------##

IF(${PACKAGE_NAME}_ENABLE_ClangFormat)
    INCLUDE(CodeFormat)
ENDIF()

##---------------------------------------------------------------------------##
## Build the documentation published on http://datatransferkit.readthedocs.org
##---------------------------------------------------------------------------##

IF(${PACKAGE_NAME}_ENABLE_ReadTheDocs)
    ADD_SUBDIRECTORY(docs)
ENDIF()

##---------------------------------------------------------------------------##
## Generate the Doxygen documentation
##---------------------------------------------------------------------------##

IF(${PACKAGE_NAME}_ENABLE_Doxygen)
    ADD_SUBDIRECTORY(docs/doxygen)
ENDIF()

##---------------------------------------------------------------------------##
## Generate a .ycm_extra_conf.py file for use with YouCompleteMe
##---------------------------------------------------------------------------##

IF(${PACKAGE_NAME}_ENABLE_YouCompleteMe)
    INCLUDE(CodeCompletion)
ENDIF()

##---------------------------------------------------------------------------##
## end of DataTransferKit/CMakeLists.txt
##---------------------------------------------------------------------------##
