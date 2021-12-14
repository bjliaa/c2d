if(NOT LZ4_LIBRARY)
  find_library(LZ4_LIBRARY_RELEASE NAMES lz4)
  find_library(LZ4_LIBRARY_DEBUG NAMES lz4d)

  include(SelectLibraryConfigurations)
  select_library_configurations(LZ4)
else()
  file(TO_CMAKE_PATH "${LZ4_LIBRARY}" LZ4_LIBRARY)
endif()

find_path(LZ4_INCLUDE_DIR NAMES lz4.h)

if(LZ4_INCLUDE_DIR AND EXISTS "${LZ4_INCLUDE_DIR}/lz4.h")
    file(STRINGS "${LZ4_INCLUDE_DIR}/lz4.h" _lz4_h_contents
      REGEX "#define LZ4_VERSION_[A-Z]+[ ]+[0-9]+")
    string(REGEX REPLACE "#define LZ4_VERSION_MAJOR[ ]+([0-9]+).+" "\\1"
      LZ4_VERSION_MAJOR "${_lz4_h_contents}")
    string(REGEX REPLACE ".+#define LZ4_VERSION_MINOR[ ]+([0-9]+).+" "\\1"
      LZ4_VERSION_MINOR "${_lz4_h_contents}")
    string(REGEX REPLACE ".+#define LZ4_VERSION_RELEASE[ ]+([0-9]+).*" "\\1"
      LZ4_VERSION_RELEASE "${_lz4_h_contents}")
    set(LZ4_VERSION "${LZ4_VERSION_MAJOR}.${LZ4_VERSION_MINOR}.${LZ4_VERSION_RELEASE}")
    unset(_lz4_h_contents)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LZ4 DEFAULT_MSG
    LZ4_LIBRARY LZ4_INCLUDE_DIR LZ4_VERSION)

mark_as_advanced(LZ4_INCLUDE_DIR LZ4_LIBRARY)

set(LZ4_LIBRARIES ${LZ4_LIBRARY})
set(LZ4_INCLUDE_DIRS ${LZ4_INCLUDE_DIR})

if(LZ4_FOUND)
  if(NOT TARGET LZ4::LZ4)
    add_library(LZ4::LZ4 UNKNOWN IMPORTED)
    set_target_properties(LZ4::LZ4 PROPERTIES
      IMPORTED_LINK_INTERFACE_LANGUAGES "C"
      INTERFACE_INCLUDE_DIRECTORIES "${LZ4_INCLUDE_DIR}")

    if(LZ4_LIBRARY_RELEASE)
      set_property(TARGET LZ4::LZ4 APPEND PROPERTY
        IMPORTED_CONFIGURATIONS RELEASE)
      set_target_properties(LZ4::LZ4 PROPERTIES
        IMPORTED_LOCATION_RELEASE "${LZ4_LIBRARY_RELEASE}")
      endif()

    if(LZ4_LIBRARY_DEBUG)
      set_property(TARGET LZ4::LZ4 APPEND PROPERTY
        IMPORTED_CONFIGURATIONS DEBUG)
      set_target_properties(LZ4::LZ4 PROPERTIES
        IMPORTED_LOCATION_DEBUG "${LZ4_LIBRARY_DEBUG}")
    endif()

    if(NOT LZ4_LIBRARY_RELEASE AND NOT LZ4_LIBRARY_DEBUG)
      set_property(TARGET LZ4::LZ4 APPEND PROPERTY
        IMPORTED_LOCATION "${LZ4_LIBRARY}")
    endif()
  endif()
endif()
