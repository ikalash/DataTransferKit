//---------------------------------------------------------------------------//
/*!
 * \file tstMeshTraitsFieldAdapter.cpp
 * \author Stuart R. Slattery
 * \brief MeshContainer unit tests.
 */
//---------------------------------------------------------------------------//

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <set>
#include <sstream>
#include <vector>

#include <DTK_FieldTraits.hpp>
#include <DTK_MeshContainer.hpp>
#include <DTK_MeshTraitsFieldAdapter.hpp>
#include <DTK_MeshTypes.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_TypeTraits.hpp>
#include <Teuchos_UnitTestHarness.hpp>

//---------------------------------------------------------------------------//
// Mesh container creation functions.
//---------------------------------------------------------------------------//
// Line segment mesh.
DataTransferKit::MeshContainer<GlobalOrdinal> buildLineContainer()
{
    using namespace DataTransferKit;

    // Make some nodes.
    Teuchos::Array<GlobalOrdinal> node_handles;
    Teuchos::Array<double> coords;

    int node_dim = 1;
    int num_nodes = 2;

    // handles
    for ( int i = 0; i < num_nodes; ++i )
    {
        node_handles.push_back( i );
    }

    // x
    coords.push_back( 0.0 );
    coords.push_back( 1.0 );

    // Make the lineahedron.
    Teuchos::Array<GlobalOrdinal> line_handles;
    Teuchos::Array<GlobalOrdinal> line_connectivity;

    // handles
    line_handles.push_back( 12 );

    // connectivity
    for ( int i = 0; i < num_nodes; ++i )
    {
        line_connectivity.push_back( i );
    }

    Teuchos::ArrayRCP<GlobalOrdinal> node_handle_array(
        node_handles.size() );
    std::copy( node_handles.begin(), node_handles.end(),
               node_handle_array.begin() );

    Teuchos::ArrayRCP<double> coords_array( coords.size() );
    std::copy( coords.begin(), coords.end(), coords_array.begin() );

    Teuchos::ArrayRCP<GlobalOrdinal> line_handle_array(
        line_handles.size() );
    std::copy( line_handles.begin(), line_handles.end(),
               line_handle_array.begin() );

    Teuchos::ArrayRCP<GlobalOrdinal> connectivity_array(
        line_connectivity.size() );
    std::copy( line_connectivity.begin(), line_connectivity.end(),
               connectivity_array.begin() );

    Teuchos::ArrayRCP<int> permutation_list( num_nodes );
    for ( int i = 0; i < permutation_list.size(); ++i )
    {
        permutation_list[i] = i;
    }

    return MeshContainer<GlobalOrdinal>(
        node_dim, node_handle_array, coords_array, DTK_LINE_SEGMENT, num_nodes,
        line_handle_array, connectivity_array, permutation_list );
}

//---------------------------------------------------------------------------//
// Tri mesh.
DataTransferKit::MeshContainer<GlobalOrdinal> buildTriContainer()
{
    using namespace DataTransferKit;

    // Make some nodes.
    Teuchos::Array<GlobalOrdinal> node_handles;
    Teuchos::Array<double> coords;

    int node_dim = 2;
    int num_nodes = 3;

    // handles
    for ( int i = 0; i < num_nodes; ++i )
    {
        node_handles.push_back( i );
    }

    // x
    coords.push_back( 0.0 );
    coords.push_back( 1.0 );
    coords.push_back( 1.0 );

    // y
    coords.push_back( 0.0 );
    coords.push_back( 0.0 );
    coords.push_back( 1.0 );

    // Make the triahedron.
    Teuchos::Array<GlobalOrdinal> tri_handles;
    Teuchos::Array<GlobalOrdinal> tri_connectivity;

    // handles
    tri_handles.push_back( 12 );

    // connectivity
    for ( int i = 0; i < num_nodes; ++i )
    {
        tri_connectivity.push_back( i );
    }

    Teuchos::ArrayRCP<GlobalOrdinal> node_handle_array(
        node_handles.size() );
    std::copy( node_handles.begin(), node_handles.end(),
               node_handle_array.begin() );

    Teuchos::ArrayRCP<double> coords_array( coords.size() );
    std::copy( coords.begin(), coords.end(), coords_array.begin() );

    Teuchos::ArrayRCP<GlobalOrdinal> tri_handle_array( tri_handles.size() );
    std::copy( tri_handles.begin(), tri_handles.end(),
               tri_handle_array.begin() );

    Teuchos::ArrayRCP<GlobalOrdinal> connectivity_array(
        tri_connectivity.size() );
    std::copy( tri_connectivity.begin(), tri_connectivity.end(),
               connectivity_array.begin() );

    Teuchos::ArrayRCP<int> permutation_list( num_nodes );
    for ( int i = 0; i < permutation_list.size(); ++i )
    {
        permutation_list[i] = i;
    }

    return MeshContainer<GlobalOrdinal>(
        node_dim, node_handle_array, coords_array, DTK_TRIANGLE, num_nodes,
        tri_handle_array, connectivity_array, permutation_list );
}

//---------------------------------------------------------------------------//
// Quad mesh.
DataTransferKit::MeshContainer<GlobalOrdinal> buildQuadContainer()
{
    using namespace DataTransferKit;

    // Make some nodes.
    Teuchos::Array<GlobalOrdinal> node_handles;
    Teuchos::Array<double> coords;

    int node_dim = 2;
    int num_nodes = 4;

    // handles
    for ( int i = 0; i < num_nodes; ++i )
    {
        node_handles.push_back( i );
    }

    // x
    coords.push_back( 0.0 );
    coords.push_back( 1.0 );
    coords.push_back( 1.0 );
    coords.push_back( 0.0 );

    // y
    coords.push_back( 0.0 );
    coords.push_back( 0.0 );
    coords.push_back( 1.0 );
    coords.push_back( 1.0 );

    // Make the quadahedron.
    Teuchos::Array<GlobalOrdinal> quad_handles;
    Teuchos::Array<GlobalOrdinal> quad_connectivity;

    // handles
    quad_handles.push_back( 12 );

    // connectivity
    for ( int i = 0; i < num_nodes; ++i )
    {
        quad_connectivity.push_back( i );
    }

    Teuchos::ArrayRCP<GlobalOrdinal> node_handle_array(
        node_handles.size() );
    std::copy( node_handles.begin(), node_handles.end(),
               node_handle_array.begin() );

    Teuchos::ArrayRCP<double> coords_array( coords.size() );
    std::copy( coords.begin(), coords.end(), coords_array.begin() );

    Teuchos::ArrayRCP<GlobalOrdinal> quad_handle_array(
        quad_handles.size() );
    std::copy( quad_handles.begin(), quad_handles.end(),
               quad_handle_array.begin() );

    Teuchos::ArrayRCP<GlobalOrdinal> connectivity_array(
        quad_connectivity.size() );
    std::copy( quad_connectivity.begin(), quad_connectivity.end(),
               connectivity_array.begin() );

    Teuchos::ArrayRCP<int> permutation_list( num_nodes );
    for ( int i = 0; i < permutation_list.size(); ++i )
    {
        permutation_list[i] = i;
    }

    return MeshContainer<GlobalOrdinal>(
        node_dim, node_handle_array, coords_array, DTK_QUADRILATERAL, num_nodes,
        quad_handle_array, connectivity_array, permutation_list );
}

//---------------------------------------------------------------------------//
// Tet mesh.
DataTransferKit::MeshContainer<GlobalOrdinal> buildTetContainer()
{
    using namespace DataTransferKit;

    // Make some nodes.
    Teuchos::Array<GlobalOrdinal> node_handles;
    Teuchos::Array<double> coords;

    int node_dim = 3;
    int num_nodes = 4;

    // handles
    for ( int i = 0; i < num_nodes; ++i )
    {
        node_handles.push_back( i );
    }

    // x
    coords.push_back( 0.0 );
    coords.push_back( 1.0 );
    coords.push_back( 1.0 );
    coords.push_back( 0.0 );

    // y
    coords.push_back( 0.0 );
    coords.push_back( 0.0 );
    coords.push_back( 1.0 );
    coords.push_back( 1.0 );

    // z
    coords.push_back( 0.0 );
    coords.push_back( 0.0 );
    coords.push_back( 0.0 );
    coords.push_back( 1.0 );

    // Make the tetahedron.
    Teuchos::Array<GlobalOrdinal> tet_handles;
    Teuchos::Array<GlobalOrdinal> tet_connectivity;

    // handles
    tet_handles.push_back( 12 );

    // connectivity
    for ( int i = 0; i < num_nodes; ++i )
    {
        tet_connectivity.push_back( i );
    }

    Teuchos::ArrayRCP<GlobalOrdinal> node_handle_array(
        node_handles.size() );
    std::copy( node_handles.begin(), node_handles.end(),
               node_handle_array.begin() );

    Teuchos::ArrayRCP<double> coords_array( coords.size() );
    std::copy( coords.begin(), coords.end(), coords_array.begin() );

    Teuchos::ArrayRCP<GlobalOrdinal> tet_handle_array( tet_handles.size() );
    std::copy( tet_handles.begin(), tet_handles.end(),
               tet_handle_array.begin() );

    Teuchos::ArrayRCP<GlobalOrdinal> connectivity_array(
        tet_connectivity.size() );
    std::copy( tet_connectivity.begin(), tet_connectivity.end(),
               connectivity_array.begin() );

    Teuchos::ArrayRCP<int> permutation_list( num_nodes );
    for ( int i = 0; i < permutation_list.size(); ++i )
    {
        permutation_list[i] = i;
    }

    return MeshContainer<GlobalOrdinal>(
        node_dim, node_handle_array, coords_array, DTK_TETRAHEDRON, num_nodes,
        tet_handle_array, connectivity_array, permutation_list );
}

//---------------------------------------------------------------------------//
// Hex mesh.
DataTransferKit::MeshContainer<GlobalOrdinal> buildHexContainer()
{
    using namespace DataTransferKit;

    // Make some nodes.
    Teuchos::Array<GlobalOrdinal> node_handles;
    Teuchos::Array<double> coords;

    int node_dim = 3;
    int num_nodes = 8;

    // handles
    for ( int i = 0; i < num_nodes; ++i )
    {
        node_handles.push_back( i );
    }

    // x
    coords.push_back( 0.0 );
    coords.push_back( 1.0 );
    coords.push_back( 1.0 );
    coords.push_back( 0.0 );
    coords.push_back( 0.0 );
    coords.push_back( 1.0 );
    coords.push_back( 1.0 );
    coords.push_back( 0.0 );

    // y
    coords.push_back( 0.0 );
    coords.push_back( 0.0 );
    coords.push_back( 1.0 );
    coords.push_back( 1.0 );
    coords.push_back( 0.0 );
    coords.push_back( 0.0 );
    coords.push_back( 1.0 );
    coords.push_back( 1.0 );

    // z
    coords.push_back( 0.0 );
    coords.push_back( 0.0 );
    coords.push_back( 0.0 );
    coords.push_back( 0.0 );
    coords.push_back( 1.0 );
    coords.push_back( 1.0 );
    coords.push_back( 1.0 );
    coords.push_back( 1.0 );

    // Make the hexahedron.
    Teuchos::Array<GlobalOrdinal> hex_handles;
    Teuchos::Array<GlobalOrdinal> hex_connectivity;

    // handles
    hex_handles.push_back( 12 );

    // connectivity
    for ( int i = 0; i < num_nodes; ++i )
    {
        hex_connectivity.push_back( i );
    }

    Teuchos::ArrayRCP<GlobalOrdinal> node_handle_array(
        node_handles.size() );
    std::copy( node_handles.begin(), node_handles.end(),
               node_handle_array.begin() );

    Teuchos::ArrayRCP<double> coords_array( coords.size() );
    std::copy( coords.begin(), coords.end(), coords_array.begin() );

    Teuchos::ArrayRCP<GlobalOrdinal> hex_handle_array( hex_handles.size() );
    std::copy( hex_handles.begin(), hex_handles.end(),
               hex_handle_array.begin() );

    Teuchos::ArrayRCP<GlobalOrdinal> connectivity_array(
        hex_connectivity.size() );
    std::copy( hex_connectivity.begin(), hex_connectivity.end(),
               connectivity_array.begin() );

    Teuchos::ArrayRCP<int> permutation_list( num_nodes );
    for ( int i = 0; i < permutation_list.size(); ++i )
    {
        permutation_list[i] = i;
    }

    return MeshContainer<GlobalOrdinal>(
        node_dim, node_handle_array, coords_array, DTK_HEXAHEDRON, num_nodes,
        hex_handle_array, connectivity_array, permutation_list );
}

//---------------------------------------------------------------------------//
// Pyramid mesh.
DataTransferKit::MeshContainer<GlobalOrdinal> buildPyramidContainer()
{
    using namespace DataTransferKit;

    // Make some nodes.
    Teuchos::Array<GlobalOrdinal> node_handles;
    Teuchos::Array<double> coords;

    int node_dim = 3;
    int num_nodes = 5;

    // handles
    for ( int i = 0; i < num_nodes; ++i )
    {
        node_handles.push_back( i );
    }

    // x
    coords.push_back( 0.0 );
    coords.push_back( 1.0 );
    coords.push_back( 1.0 );
    coords.push_back( 0.0 );
    coords.push_back( 0.5 );

    // y
    coords.push_back( 0.0 );
    coords.push_back( 0.0 );
    coords.push_back( 1.0 );
    coords.push_back( 1.0 );
    coords.push_back( 0.5 );

    // z
    coords.push_back( 0.0 );
    coords.push_back( 0.0 );
    coords.push_back( 0.0 );
    coords.push_back( 0.0 );
    coords.push_back( 1.0 );

    // Make the pyramidahedron.
    Teuchos::Array<GlobalOrdinal> pyramid_handles;
    Teuchos::Array<GlobalOrdinal> pyramid_connectivity;

    // handles
    pyramid_handles.push_back( 12 );

    // connectivity
    for ( int i = 0; i < num_nodes; ++i )
    {
        pyramid_connectivity.push_back( i );
    }

    Teuchos::ArrayRCP<GlobalOrdinal> node_handle_array(
        node_handles.size() );
    std::copy( node_handles.begin(), node_handles.end(),
               node_handle_array.begin() );

    Teuchos::ArrayRCP<double> coords_array( coords.size() );
    std::copy( coords.begin(), coords.end(), coords_array.begin() );

    Teuchos::ArrayRCP<GlobalOrdinal> pyramid_handle_array(
        pyramid_handles.size() );
    std::copy( pyramid_handles.begin(), pyramid_handles.end(),
               pyramid_handle_array.begin() );

    Teuchos::ArrayRCP<GlobalOrdinal> connectivity_array(
        pyramid_connectivity.size() );
    std::copy( pyramid_connectivity.begin(), pyramid_connectivity.end(),
               connectivity_array.begin() );

    Teuchos::ArrayRCP<int> permutation_list( num_nodes );
    for ( int i = 0; i < permutation_list.size(); ++i )
    {
        permutation_list[i] = i;
    }

    return MeshContainer<GlobalOrdinal>(
        node_dim, node_handle_array, coords_array, DTK_PYRAMID, num_nodes,
        pyramid_handle_array, connectivity_array, permutation_list );
}
//---------------------------------------------------------------------------//
// Wedge mesh.
DataTransferKit::MeshContainer<GlobalOrdinal> buildWedgeContainer()
{
    using namespace DataTransferKit;

    // Make some nodes.
    Teuchos::Array<GlobalOrdinal> node_handles;
    Teuchos::Array<double> coords;

    int node_dim = 3;
    int num_nodes = 6;

    // handles
    for ( int i = 0; i < num_nodes; ++i )
    {
        node_handles.push_back( i );
    }

    // x
    coords.push_back( 0.0 );
    coords.push_back( 1.0 );
    coords.push_back( 1.0 );
    coords.push_back( 0.0 );
    coords.push_back( 1.0 );
    coords.push_back( 1.0 );

    // y
    coords.push_back( 0.0 );
    coords.push_back( 0.0 );
    coords.push_back( 1.0 );
    coords.push_back( 0.0 );
    coords.push_back( 0.0 );
    coords.push_back( 1.0 );

    // z
    coords.push_back( 0.0 );
    coords.push_back( 0.0 );
    coords.push_back( 0.0 );
    coords.push_back( 1.0 );
    coords.push_back( 1.0 );
    coords.push_back( 1.0 );

    // Make the wedge.
    Teuchos::Array<GlobalOrdinal> wedge_handles;
    Teuchos::Array<GlobalOrdinal> wedge_connectivity;

    // handles
    wedge_handles.push_back( 12 );

    // connectivity
    for ( int i = 0; i < num_nodes; ++i )
    {
        wedge_connectivity.push_back( i );
    }

    Teuchos::ArrayRCP<GlobalOrdinal> node_handle_array(
        node_handles.size() );
    std::copy( node_handles.begin(), node_handles.end(),
               node_handle_array.begin() );

    Teuchos::ArrayRCP<double> coords_array( coords.size() );
    std::copy( coords.begin(), coords.end(), coords_array.begin() );

    Teuchos::ArrayRCP<GlobalOrdinal> wedge_handle_array(
        wedge_handles.size() );
    std::copy( wedge_handles.begin(), wedge_handles.end(),
               wedge_handle_array.begin() );

    Teuchos::ArrayRCP<GlobalOrdinal> connectivity_array(
        wedge_connectivity.size() );
    std::copy( wedge_connectivity.begin(), wedge_connectivity.end(),
               connectivity_array.begin() );

    Teuchos::ArrayRCP<int> permutation_list( num_nodes );
    for ( int i = 0; i < permutation_list.size(); ++i )
    {
        permutation_list[i] = i;
    }

    return MeshContainer<GlobalOrdinal>(
        node_dim, node_handle_array, coords_array, DTK_WEDGE, num_nodes,
        wedge_handle_array, connectivity_array, permutation_list );
}

//---------------------------------------------------------------------------//
// Tests
//---------------------------------------------------------------------------//
// Line mesh.
TEUCHOS_UNIT_TEST( FieldTraits, line_adapter_test )
{
    using namespace DataTransferKit;

    // Create a mesh container.
    typedef FieldTraits<MeshContainer<GlobalOrdinal>> FT;
    MeshContainer<GlobalOrdinal> mesh_container = buildLineContainer();

    // Mesh parameters.
    int node_dim = 1;
    int num_nodes = 2;

    // Basic container info.
    TEST_ASSERT( node_dim == (int)FT::dim( mesh_container ) );
    TEST_ASSERT( node_dim * num_nodes == (int)FT::size( mesh_container ) );
    TEST_ASSERT( !FT::empty( mesh_container ) );

    // Coords.
    for ( int i = 0; i < num_nodes; ++i )
    {
        TEST_ASSERT( mesh_container.coordsBegin()[i] ==
                     FT::begin( mesh_container )[i] );
    }
}

//---------------------------------------------------------------------------//
// Tri mesh.
TEUCHOS_UNIT_TEST( FieldTraits, tri_adapter_test )
{
    using namespace DataTransferKit;

    // Create a mesh container.
    typedef FieldTraits<MeshContainer<GlobalOrdinal>> FT;
    MeshContainer<GlobalOrdinal> mesh_container = buildTriContainer();

    // Mesh parameters.
    int node_dim = 2;
    int num_nodes = 3;

    // Basic container info.
    TEST_ASSERT( node_dim == (int)FT::dim( mesh_container ) );
    TEST_ASSERT( node_dim * num_nodes == (int)FT::size( mesh_container ) );
    TEST_ASSERT( !FT::empty( mesh_container ) );

    // Coords.
    for ( int i = 0; i < num_nodes; ++i )
    {
        TEST_ASSERT( mesh_container.coordsBegin()[i] ==
                     FT::begin( mesh_container )[i] );
    }
}

//---------------------------------------------------------------------------//
// Quad mesh.
TEUCHOS_UNIT_TEST( FieldTraits, quad_adapter_test )
{
    using namespace DataTransferKit;

    // Create a mesh container.
    typedef FieldTraits<MeshContainer<GlobalOrdinal>> FT;
    MeshContainer<GlobalOrdinal> mesh_container = buildQuadContainer();

    // Mesh parameters.
    int node_dim = 2;
    int num_nodes = 4;

    // Basic container info.
    TEST_ASSERT( node_dim == (int)FT::dim( mesh_container ) );
    TEST_ASSERT( node_dim * num_nodes == (int)FT::size( mesh_container ) );
    TEST_ASSERT( !FT::empty( mesh_container ) );

    // Coords.
    for ( int i = 0; i < num_nodes; ++i )
    {
        TEST_ASSERT( mesh_container.coordsBegin()[i] ==
                     FT::begin( mesh_container )[i] );
    }
}

//---------------------------------------------------------------------------//
// Tet mesh.
TEUCHOS_UNIT_TEST( FieldTraits, tet_adapter_test )
{
    using namespace DataTransferKit;

    // Create a mesh container.
    typedef FieldTraits<MeshContainer<GlobalOrdinal>> FT;
    MeshContainer<GlobalOrdinal> mesh_container = buildTetContainer();

    // Mesh parameters.
    int node_dim = 3;
    int num_nodes = 4;

    // Basic container info.
    TEST_ASSERT( node_dim == (int)FT::dim( mesh_container ) );
    TEST_ASSERT( node_dim * num_nodes == (int)FT::size( mesh_container ) );
    TEST_ASSERT( !FT::empty( mesh_container ) );

    // Coords.
    for ( int i = 0; i < num_nodes; ++i )
    {
        TEST_ASSERT( mesh_container.coordsBegin()[i] ==
                     FT::begin( mesh_container )[i] );
    }
}

//---------------------------------------------------------------------------//
// Hex mesh.
TEUCHOS_UNIT_TEST( FieldTraits, hex_adapter_test )
{
    using namespace DataTransferKit;

    // Create a mesh container.
    typedef FieldTraits<MeshContainer<GlobalOrdinal>> FT;
    MeshContainer<GlobalOrdinal> mesh_container = buildHexContainer();

    // Mesh parameters.
    int node_dim = 3;
    int num_nodes = 8;

    // Basic container info.
    TEST_ASSERT( node_dim == (int)FT::dim( mesh_container ) );
    TEST_ASSERT( node_dim * num_nodes == (int)FT::size( mesh_container ) );
    TEST_ASSERT( !FT::empty( mesh_container ) );

    // Coords.
    for ( int i = 0; i < num_nodes; ++i )
    {
        TEST_ASSERT( mesh_container.coordsBegin()[i] ==
                     FT::begin( mesh_container )[i] );
    }
}

//---------------------------------------------------------------------------//
// Pyramid mesh.
TEUCHOS_UNIT_TEST( FieldTraits, pyramid_adapter_test )
{
    using namespace DataTransferKit;

    // Create a mesh container.
    typedef FieldTraits<MeshContainer<GlobalOrdinal>> FT;
    MeshContainer<GlobalOrdinal> mesh_container = buildPyramidContainer();

    // Mesh parameters.
    int node_dim = 3;
    int num_nodes = 5;

    // Basic container info.
    TEST_ASSERT( node_dim == (int)FT::dim( mesh_container ) );
    TEST_ASSERT( node_dim * num_nodes == (int)FT::size( mesh_container ) );
    TEST_ASSERT( !FT::empty( mesh_container ) );

    // Coords.
    for ( int i = 0; i < num_nodes; ++i )
    {
        TEST_ASSERT( mesh_container.coordsBegin()[i] ==
                     FT::begin( mesh_container )[i] );
    }
}

//---------------------------------------------------------------------------//
// Wedge mesh.
TEUCHOS_UNIT_TEST( FieldTraits, wedge_adapter_test )
{
    using namespace DataTransferKit;

    // Create a mesh container.
    typedef FieldTraits<MeshContainer<GlobalOrdinal>> FT;
    MeshContainer<GlobalOrdinal> mesh_container = buildWedgeContainer();

    // Mesh parameters.
    int node_dim = 3;
    int num_nodes = 6;

    // Basic container info.
    TEST_ASSERT( node_dim == (int)FT::dim( mesh_container ) );
    TEST_ASSERT( node_dim * num_nodes == (int)FT::size( mesh_container ) );
    TEST_ASSERT( !FT::empty( mesh_container ) );

    // Coords.
    for ( int i = 0; i < num_nodes; ++i )
    {
        TEST_ASSERT( mesh_container.coordsBegin()[i] ==
                     FT::begin( mesh_container )[i] );
    }
}

//---------------------------------------------------------------------------//
// end tstMeshTraitsFieldAdapter.cpp
//---------------------------------------------------------------------------//
