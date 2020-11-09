//---------------------------------------------------------------------------//
/*!
 * \file tstMeshContainer.cpp
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

#include <DTK_MeshContainer.hpp>
#include <DTK_MeshTraits.hpp>
#include <DTK_MeshTypes.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_TypeTraits.hpp>
#include <Teuchos_UnitTestHarness.hpp>


//---------------------------------------------------------------------------//
// MPI Setup
//---------------------------------------------------------------------------//

template <class Ordinal>
Teuchos::RCP<const Teuchos::Comm<Ordinal>> getDefaultComm()
{
#ifdef HAVE_MPI
    return Teuchos::DefaultComm<Ordinal>::getComm();
#else
    return Teuchos::rcp( new Teuchos::SerialComm<Ordinal>() );
#endif
}

//---------------------------------------------------------------------------//
// Mesh container creation functions.
//---------------------------------------------------------------------------//
// Line segment mesh.
DataTransferKit::MeshContainer<GlobalOrdinal> buildLineContainer()
{
    using namespace DataTransferKit;

    // Make some vertices.
    Teuchos::Array<GlobalOrdinal> vertex_handles;
    Teuchos::Array<double> coords;

    int vertex_dim = 1;
    unsigned int num_vertices = 2;

    // handles
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        vertex_handles.push_back( i );
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
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        line_connectivity.push_back( i );
    }

    Teuchos::ArrayRCP<GlobalOrdinal> vertex_handle_array(
        vertex_handles.size() );
    std::copy( vertex_handles.begin(), vertex_handles.end(),
               vertex_handle_array.begin() );

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

    Teuchos::ArrayRCP<int> permutation_list( num_vertices );
    for ( unsigned int i = 0; i < permutation_list.size(); ++i )
    {
        permutation_list[i] = i;
    }

    return MeshContainer<GlobalOrdinal>(
        vertex_dim, vertex_handle_array, coords_array, DTK_LINE_SEGMENT,
        num_vertices, line_handle_array, connectivity_array, permutation_list );
}

//---------------------------------------------------------------------------//
// Tri mesh.
DataTransferKit::MeshContainer<GlobalOrdinal> buildTriContainer()
{
    using namespace DataTransferKit;

    // Make some vertices.
    Teuchos::Array<GlobalOrdinal> vertex_handles;
    Teuchos::Array<double> coords;

    int vertex_dim = 2;
    unsigned int num_vertices = 3;

    // handles
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        vertex_handles.push_back( i );
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
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        tri_connectivity.push_back( i );
    }

    Teuchos::ArrayRCP<GlobalOrdinal> vertex_handle_array(
        vertex_handles.size() );
    std::copy( vertex_handles.begin(), vertex_handles.end(),
               vertex_handle_array.begin() );

    Teuchos::ArrayRCP<double> coords_array( coords.size() );
    std::copy( coords.begin(), coords.end(), coords_array.begin() );

    Teuchos::ArrayRCP<GlobalOrdinal> tri_handle_array( tri_handles.size() );
    std::copy( tri_handles.begin(), tri_handles.end(),
               tri_handle_array.begin() );

    Teuchos::ArrayRCP<GlobalOrdinal> connectivity_array(
        tri_connectivity.size() );
    std::copy( tri_connectivity.begin(), tri_connectivity.end(),
               connectivity_array.begin() );

    Teuchos::ArrayRCP<int> permutation_list( num_vertices );
    for ( unsigned int i = 0; i < permutation_list.size(); ++i )
    {
        permutation_list[i] = i;
    }

    return MeshContainer<GlobalOrdinal>(
        vertex_dim, vertex_handle_array, coords_array, DTK_TRIANGLE,
        num_vertices, tri_handle_array, connectivity_array, permutation_list );
}

//---------------------------------------------------------------------------//
// Quad mesh.
DataTransferKit::MeshContainer<GlobalOrdinal> buildQuadContainer()
{
    using namespace DataTransferKit;

    // Make some vertices.
    Teuchos::Array<GlobalOrdinal> vertex_handles;
    Teuchos::Array<double> coords;

    int vertex_dim = 2;
    unsigned int num_vertices = 4;

    // handles
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        vertex_handles.push_back( i );
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
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        quad_connectivity.push_back( i );
    }

    Teuchos::ArrayRCP<GlobalOrdinal> vertex_handle_array(
        vertex_handles.size() );
    std::copy( vertex_handles.begin(), vertex_handles.end(),
               vertex_handle_array.begin() );

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

    Teuchos::ArrayRCP<int> permutation_list( num_vertices );
    for ( unsigned int i = 0; i < permutation_list.size(); ++i )
    {
        permutation_list[i] = i;
    }

    return MeshContainer<GlobalOrdinal>(
        vertex_dim, vertex_handle_array, coords_array, DTK_QUADRILATERAL,
        num_vertices, quad_handle_array, connectivity_array, permutation_list );
}

//---------------------------------------------------------------------------//
// Tet mesh.
DataTransferKit::MeshContainer<GlobalOrdinal> buildTetContainer()
{
    using namespace DataTransferKit;

    // Make some vertices.
    Teuchos::Array<GlobalOrdinal> vertex_handles;
    Teuchos::Array<double> coords;

    int vertex_dim = 3;
    unsigned int num_vertices = 4;

    // handles
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        vertex_handles.push_back( i );
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
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        tet_connectivity.push_back( i );
    }

    Teuchos::ArrayRCP<GlobalOrdinal> vertex_handle_array(
        vertex_handles.size() );
    std::copy( vertex_handles.begin(), vertex_handles.end(),
               vertex_handle_array.begin() );

    Teuchos::ArrayRCP<double> coords_array( coords.size() );
    std::copy( coords.begin(), coords.end(), coords_array.begin() );

    Teuchos::ArrayRCP<GlobalOrdinal> tet_handle_array( tet_handles.size() );
    std::copy( tet_handles.begin(), tet_handles.end(),
               tet_handle_array.begin() );

    Teuchos::ArrayRCP<GlobalOrdinal> connectivity_array(
        tet_connectivity.size() );
    std::copy( tet_connectivity.begin(), tet_connectivity.end(),
               connectivity_array.begin() );

    Teuchos::ArrayRCP<int> permutation_list( num_vertices );
    for ( unsigned int i = 0; i < permutation_list.size(); ++i )
    {
        permutation_list[i] = i;
    }

    return MeshContainer<GlobalOrdinal>(
        vertex_dim, vertex_handle_array, coords_array, DTK_TETRAHEDRON,
        num_vertices, tet_handle_array, connectivity_array, permutation_list );
}

//---------------------------------------------------------------------------//
// Hex mesh.
DataTransferKit::MeshContainer<GlobalOrdinal> buildHexContainer()
{
    using namespace DataTransferKit;

    // Make some vertices.
    Teuchos::Array<GlobalOrdinal> vertex_handles;
    Teuchos::Array<double> coords;

    int vertex_dim = 3;
    unsigned int num_vertices = 8;

    // handles
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        vertex_handles.push_back( i );
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
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        hex_connectivity.push_back( i );
    }

    Teuchos::ArrayRCP<GlobalOrdinal> vertex_handle_array(
        vertex_handles.size() );
    std::copy( vertex_handles.begin(), vertex_handles.end(),
               vertex_handle_array.begin() );

    Teuchos::ArrayRCP<double> coords_array( coords.size() );
    std::copy( coords.begin(), coords.end(), coords_array.begin() );

    Teuchos::ArrayRCP<GlobalOrdinal> hex_handle_array( hex_handles.size() );
    std::copy( hex_handles.begin(), hex_handles.end(),
               hex_handle_array.begin() );

    Teuchos::ArrayRCP<GlobalOrdinal> connectivity_array(
        hex_connectivity.size() );
    std::copy( hex_connectivity.begin(), hex_connectivity.end(),
               connectivity_array.begin() );

    Teuchos::ArrayRCP<int> permutation_list( num_vertices );
    for ( unsigned int i = 0; i < permutation_list.size(); ++i )
    {
        permutation_list[i] = i;
    }

    return MeshContainer<GlobalOrdinal>(
        vertex_dim, vertex_handle_array, coords_array, DTK_HEXAHEDRON,
        num_vertices, hex_handle_array, connectivity_array, permutation_list );
}

//---------------------------------------------------------------------------//
// Pyramid mesh.
DataTransferKit::MeshContainer<GlobalOrdinal> buildPyramidContainer()
{
    using namespace DataTransferKit;

    // Make some vertices.
    Teuchos::Array<GlobalOrdinal> vertex_handles;
    Teuchos::Array<double> coords;

    int vertex_dim = 3;
    unsigned int num_vertices = 5;

    // handles
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        vertex_handles.push_back( i );
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
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        pyramid_connectivity.push_back( i );
    }

    Teuchos::ArrayRCP<GlobalOrdinal> vertex_handle_array(
        vertex_handles.size() );
    std::copy( vertex_handles.begin(), vertex_handles.end(),
               vertex_handle_array.begin() );

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

    Teuchos::ArrayRCP<int> permutation_list( num_vertices );
    for ( unsigned int i = 0; i < permutation_list.size(); ++i )
    {
        permutation_list[i] = i;
    }

    return MeshContainer<GlobalOrdinal>(
        vertex_dim, vertex_handle_array, coords_array, DTK_PYRAMID,
        num_vertices, pyramid_handle_array, connectivity_array,
        permutation_list );
}

//---------------------------------------------------------------------------//
// Wedge mesh.
DataTransferKit::MeshContainer<GlobalOrdinal> buildWedgeContainer()
{
    using namespace DataTransferKit;

    // Make some vertices.
    Teuchos::Array<GlobalOrdinal> vertex_handles;
    Teuchos::Array<double> coords;

    int vertex_dim = 3;
    unsigned int num_vertices = 6;

    // handles
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        vertex_handles.push_back( i );
    }

    // x
    coords.push_back( 0.0 );
    coords.push_back( 1.0 );
    coords.push_back( 0.5 );
    coords.push_back( 0.0 );
    coords.push_back( 1.0 );
    coords.push_back( 0.5 );

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
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        wedge_connectivity.push_back( i );
    }

    Teuchos::ArrayRCP<GlobalOrdinal> vertex_handle_array(
        vertex_handles.size() );
    std::copy( vertex_handles.begin(), vertex_handles.end(),
               vertex_handle_array.begin() );

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

    Teuchos::ArrayRCP<int> permutation_list( num_vertices );
    for ( unsigned int i = 0; i < permutation_list.size(); ++i )
    {
        permutation_list[i] = i;
    }

    return MeshContainer<GlobalOrdinal>(
        vertex_dim, vertex_handle_array, coords_array, DTK_WEDGE, num_vertices,
        wedge_handle_array, connectivity_array, permutation_list );
}

//---------------------------------------------------------------------------//
// Tests
//---------------------------------------------------------------------------//
// Line mesh.
TEUCHOS_UNIT_TEST( MeshContainer, line_container_test )
{
    using namespace DataTransferKit;

    // Create a mesh container.
    typedef MeshTraits<MeshContainer<GlobalOrdinal>> MT;
    MeshContainer<GlobalOrdinal> mesh_container = buildLineContainer();

    // Mesh parameters.
    int vertex_dim = 1;
    unsigned int num_vertices = 2;
    int element_topo = DTK_LINE_SEGMENT;

    // Basic container info.
    TEST_EQUALITY( (int)MT::vertexDim( mesh_container ), vertex_dim );
    TEST_EQUALITY( (int)mesh_container.getVertexDim(), vertex_dim );
    TEST_EQUALITY( MT::verticesPerElement( mesh_container ),
                   (int)num_vertices );
    TEST_EQUALITY( mesh_container.getVerticesPerElement(), (int)num_vertices );
    TEST_EQUALITY( (int)MT::elementTopology( mesh_container ), element_topo );
    TEST_EQUALITY( (int)mesh_container.getElementTopology(), element_topo );

    // Vertices.
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        TEST_EQUALITY( *( MT::verticesBegin( mesh_container ) + i ), i );
        TEST_EQUALITY( *( mesh_container.verticesBegin() + i ), i );
    }

    // Coords.
    // x
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[0], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[1], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[0], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[1], 1.0 );

    // Elements.
    TEST_EQUALITY( *MT::elementsBegin( mesh_container ), 12 );
    TEST_EQUALITY( *mesh_container.elementsBegin(), 12 );

    // Connectivity.
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        TEST_EQUALITY( *( MT::connectivityBegin( mesh_container ) + i ), i );
        TEST_EQUALITY( *( mesh_container.connectivityBegin() + i ), i );
    }

    // Permutation.
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        TEST_EQUALITY( *( MT::permutationBegin( mesh_container ) + i ),
                       (int)i );
        TEST_EQUALITY( *( mesh_container.permutationBegin() + i ), (int)i );
    }
}

//---------------------------------------------------------------------------//
// Tri mesh.
TEUCHOS_UNIT_TEST( MeshContainer, tri_container_test )
{
    using namespace DataTransferKit;

    // Create a mesh container.
    typedef MeshTraits<MeshContainer<GlobalOrdinal>> MT;
    MeshContainer<GlobalOrdinal> mesh_container = buildTriContainer();

    // Mesh parameters.
    int vertex_dim = 2;
    unsigned int num_vertices = 3;
    int element_topo = DTK_TRIANGLE;

    // Basic container info.
    TEST_EQUALITY( (int)MT::vertexDim( mesh_container ), vertex_dim );
    TEST_EQUALITY( (int)mesh_container.getVertexDim(), vertex_dim );
    TEST_EQUALITY( MT::verticesPerElement( mesh_container ),
                   (int)num_vertices );
    TEST_EQUALITY( mesh_container.getVerticesPerElement(), (int)num_vertices );
    TEST_EQUALITY( (int)MT::elementTopology( mesh_container ), element_topo );
    TEST_EQUALITY( (int)mesh_container.getElementTopology(), element_topo );

    // Vertices.
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        TEST_EQUALITY( *( MT::verticesBegin( mesh_container ) + i ), i );
        TEST_EQUALITY( *( mesh_container.verticesBegin() + i ), i );
    }

    // Coords.
    // x
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[0], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[1], 1.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[2], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[0], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[1], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[2], 1.0 );

    // y
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[3], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[4], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[5], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[3], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[4], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[5], 1.0 );

    // Elements.
    TEST_EQUALITY( *MT::elementsBegin( mesh_container ), 12 );
    TEST_EQUALITY( *mesh_container.elementsBegin(), 12 );

    // Connectivity.
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        TEST_EQUALITY( *( MT::connectivityBegin( mesh_container ) + i ), i );
        TEST_EQUALITY( *( mesh_container.connectivityBegin() + i ), i );
    }

    // Permutation.
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        TEST_EQUALITY( *( MT::permutationBegin( mesh_container ) + i ),
                       (int)i );
        TEST_EQUALITY( *( mesh_container.permutationBegin() + i ), (int)i );
    }
}

//---------------------------------------------------------------------------//
// Quad mesh.
TEUCHOS_UNIT_TEST( MeshContainer, quad_container_test )
{
    using namespace DataTransferKit;

    // Create a mesh container.
    typedef MeshTraits<MeshContainer<GlobalOrdinal>> MT;
    MeshContainer<GlobalOrdinal> mesh_container = buildQuadContainer();

    // Mesh parameters.
    int vertex_dim = 2;
    unsigned int num_vertices = 4;
    int element_topo = DTK_QUADRILATERAL;

    // Basic container info.
    TEST_EQUALITY( (int)MT::vertexDim( mesh_container ), vertex_dim );
    TEST_EQUALITY( (int)mesh_container.getVertexDim(), vertex_dim );
    TEST_EQUALITY( MT::verticesPerElement( mesh_container ),
                   (int)num_vertices );
    TEST_EQUALITY( mesh_container.getVerticesPerElement(), (int)num_vertices );
    TEST_EQUALITY( (int)MT::elementTopology( mesh_container ), element_topo );
    TEST_EQUALITY( (int)mesh_container.getElementTopology(), element_topo );

    // Vertices.
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        TEST_EQUALITY( *( MT::verticesBegin( mesh_container ) + i ), i );
        TEST_EQUALITY( *( mesh_container.verticesBegin() + i ), i );
    }

    // Coords.
    // x
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[0], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[1], 1.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[2], 1.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[3], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[0], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[1], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[2], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[3], 0.0 );

    // y
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[4], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[5], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[6], 1.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[7], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[4], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[5], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[6], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[7], 1.0 );

    // Elements.
    TEST_EQUALITY( *MT::elementsBegin( mesh_container ), 12 );
    TEST_EQUALITY( *mesh_container.elementsBegin(), 12 );

    // Connectivity.
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        TEST_EQUALITY( *( MT::connectivityBegin( mesh_container ) + i ), i );
        TEST_EQUALITY( *( mesh_container.connectivityBegin() + i ), i );
    }

    // Permutation.
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        TEST_EQUALITY( *( MT::permutationBegin( mesh_container ) + i ),
                       (int)i );
        TEST_EQUALITY( *( mesh_container.permutationBegin() + i ), (int)i );
    }
}

//---------------------------------------------------------------------------//
// Tet mesh.
TEUCHOS_UNIT_TEST( MeshContainer, tet_container_test )
{
    using namespace DataTransferKit;

    // Create a mesh container.
    typedef MeshTraits<MeshContainer<GlobalOrdinal>> MT;
    MeshContainer<GlobalOrdinal> mesh_container = buildTetContainer();

    // Mesh parameters.
    int vertex_dim = 3;
    unsigned int num_vertices = 4;
    int element_topo = DTK_TETRAHEDRON;

    // Basic container info.
    TEST_EQUALITY( (int)MT::vertexDim( mesh_container ), vertex_dim );
    TEST_EQUALITY( (int)mesh_container.getVertexDim(), vertex_dim );
    TEST_EQUALITY( MT::verticesPerElement( mesh_container ),
                   (int)num_vertices );
    TEST_EQUALITY( mesh_container.getVerticesPerElement(), (int)num_vertices );
    TEST_EQUALITY( (int)MT::elementTopology( mesh_container ), element_topo );
    TEST_EQUALITY( (int)mesh_container.getElementTopology(), element_topo );

    // Vertices.
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        TEST_EQUALITY( *( MT::verticesBegin( mesh_container ) + i ), i );
        TEST_EQUALITY( *( mesh_container.verticesBegin() + i ), i );
    }

    // Coords.
    // x
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[0], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[1], 1.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[2], 1.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[3], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[0], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[1], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[2], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[3], 0.0 );

    // y
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[4], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[5], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[6], 1.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[7], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[4], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[5], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[6], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[7], 1.0 );

    // z
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[8], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[9], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[10], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[11], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[8], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[9], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[10], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[11], 1.0 );

    // Elements.
    TEST_EQUALITY( *MT::elementsBegin( mesh_container ), 12 );
    TEST_EQUALITY( *mesh_container.elementsBegin(), 12 );

    // Connectivity.
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        TEST_EQUALITY( *( MT::connectivityBegin( mesh_container ) + i ), i );
        TEST_EQUALITY( *( mesh_container.connectivityBegin() + i ), i );
    }

    // Permutation.
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        TEST_EQUALITY( *( MT::permutationBegin( mesh_container ) + i ),
                       (int)i );
        TEST_EQUALITY( *( mesh_container.permutationBegin() + i ), (int)i );
    }
}

//---------------------------------------------------------------------------//
// Hex mesh.
TEUCHOS_UNIT_TEST( MeshContainer, hex_container_test )
{
    using namespace DataTransferKit;

    // Create a mesh container.
    typedef MeshTraits<MeshContainer<GlobalOrdinal>> MT;
    MeshContainer<GlobalOrdinal> mesh_container = buildHexContainer();

    // Mesh parameters.
    int vertex_dim = 3;
    unsigned int num_vertices = 8;
    int element_topo = DTK_HEXAHEDRON;

    // Basic container info.
    TEST_EQUALITY( (int)MT::vertexDim( mesh_container ), vertex_dim );
    TEST_EQUALITY( (int)mesh_container.getVertexDim(), vertex_dim );
    TEST_EQUALITY( MT::verticesPerElement( mesh_container ),
                   (int)num_vertices );
    TEST_EQUALITY( mesh_container.getVerticesPerElement(), (int)num_vertices );
    TEST_EQUALITY( (int)MT::elementTopology( mesh_container ), element_topo );
    TEST_EQUALITY( (int)mesh_container.getElementTopology(), element_topo );

    // Vertices.
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        TEST_EQUALITY( *( MT::verticesBegin( mesh_container ) + i ), i );
        TEST_EQUALITY( *( mesh_container.verticesBegin() + i ), i );
    }

    // Coords.
    // x
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[0], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[1], 1.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[2], 1.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[3], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[4], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[5], 1.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[6], 1.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[7], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[0], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[1], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[2], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[3], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[4], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[5], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[6], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[7], 0.0 );

    // y
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[8], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[9], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[10], 1.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[11], 1.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[12], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[13], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[14], 1.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[15], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[8], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[9], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[10], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[11], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[12], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[13], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[14], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[15], 1.0 );

    // z
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[16], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[17], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[18], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[19], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[20], 1.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[21], 1.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[22], 1.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[23], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[16], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[17], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[18], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[19], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[20], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[21], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[22], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[23], 1.0 );

    // Elements.
    TEST_EQUALITY( *MT::elementsBegin( mesh_container ), 12 );
    TEST_EQUALITY( *mesh_container.elementsBegin(), 12 );

    // Connectivity.
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        TEST_EQUALITY( *( MT::connectivityBegin( mesh_container ) + i ), i );
        TEST_EQUALITY( *( mesh_container.connectivityBegin() + i ), i );
    }

    // Permutation.
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        TEST_EQUALITY( *( MT::permutationBegin( mesh_container ) + i ),
                       (int)i );
        TEST_EQUALITY( *( mesh_container.permutationBegin() + i ), (int)i );
    }
}

//---------------------------------------------------------------------------//
// Pyramid mesh.
TEUCHOS_UNIT_TEST( MeshContainer, pyramid_container_test )
{
    using namespace DataTransferKit;

    // Create a mesh container.
    typedef MeshTraits<MeshContainer<GlobalOrdinal>> MT;
    MeshContainer<GlobalOrdinal> mesh_container = buildPyramidContainer();

    // Mesh parameters.
    int vertex_dim = 3;
    unsigned int num_vertices = 5;
    int element_topo = DTK_PYRAMID;

    // Basic container info.
    TEST_EQUALITY( (int)MT::vertexDim( mesh_container ), vertex_dim );
    TEST_EQUALITY( (int)mesh_container.getVertexDim(), vertex_dim );
    TEST_EQUALITY( MT::verticesPerElement( mesh_container ),
                   (int)num_vertices );
    TEST_EQUALITY( mesh_container.getVerticesPerElement(), (int)num_vertices );
    TEST_EQUALITY( (int)MT::elementTopology( mesh_container ), element_topo );
    TEST_EQUALITY( (int)mesh_container.getElementTopology(), element_topo );

    // Vertices.
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        TEST_EQUALITY( *( MT::verticesBegin( mesh_container ) + i ), i );
        TEST_EQUALITY( *( mesh_container.verticesBegin() + i ), i );
    }

    // Coords.
    // x
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[0], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[1], 1.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[2], 1.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[3], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[4], 0.5 );
    TEST_EQUALITY( mesh_container.coordsBegin()[0], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[1], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[2], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[3], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[4], 0.5 );

    // y
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[5], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[6], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[7], 1.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[8], 1.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[9], 0.5 );
    TEST_EQUALITY( mesh_container.coordsBegin()[5], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[6], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[7], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[8], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[9], 0.5 );

    // z
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[10], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[11], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[12], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[13], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[14], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[10], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[11], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[12], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[13], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[14], 1.0 );

    // Elements.
    TEST_EQUALITY( *MT::elementsBegin( mesh_container ), 12 );
    TEST_EQUALITY( *mesh_container.elementsBegin(), 12 );

    // Connectivity.
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        TEST_EQUALITY( *( MT::connectivityBegin( mesh_container ) + i ), i );
        TEST_EQUALITY( *( mesh_container.connectivityBegin() + i ), i );
    }

    // Permutation.
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        TEST_EQUALITY( *( MT::permutationBegin( mesh_container ) + i ),
                       (int)i );
        TEST_EQUALITY( *( mesh_container.permutationBegin() + i ), (int)i );
    }
}

//---------------------------------------------------------------------------//
// Wedge mesh.
TEUCHOS_UNIT_TEST( MeshContainer, wedge_container_test )
{
    using namespace DataTransferKit;

    // Create a mesh container.
    typedef MeshTraits<MeshContainer<GlobalOrdinal>> MT;
    MeshContainer<GlobalOrdinal> mesh_container = buildWedgeContainer();

    // Mesh parameters.
    int vertex_dim = 3;
    unsigned int num_vertices = 6;
    int element_topo = DTK_WEDGE;

    // Basic container info.
    TEST_EQUALITY( (int)MT::vertexDim( mesh_container ), vertex_dim );
    TEST_EQUALITY( (int)mesh_container.getVertexDim(), vertex_dim );
    TEST_EQUALITY( MT::verticesPerElement( mesh_container ),
                   (int)num_vertices );
    TEST_EQUALITY( mesh_container.getVerticesPerElement(), (int)num_vertices );
    TEST_EQUALITY( (int)MT::elementTopology( mesh_container ), element_topo );
    TEST_EQUALITY( (int)mesh_container.getElementTopology(), element_topo );

    // Vertices.
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        TEST_EQUALITY( *( MT::verticesBegin( mesh_container ) + i ), i );
        TEST_EQUALITY( *( mesh_container.verticesBegin() + i ), i );
    }

    // Coords.
    // x
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[0], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[1], 1.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[2], 0.5 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[3], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[4], 1.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[5], 0.5 );
    TEST_EQUALITY( mesh_container.coordsBegin()[0], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[1], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[2], 0.5 );
    TEST_EQUALITY( mesh_container.coordsBegin()[3], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[4], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[5], 0.5 );

    // y
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[6], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[7], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[8], 1.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[9], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[10], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[11], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[6], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[7], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[8], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[9], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[10], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[11], 1.0 );

    // z
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[12], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[13], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[14], 0.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[15], 1.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[16], 1.0 );
    TEST_EQUALITY( MT::coordsBegin( mesh_container )[17], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[12], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[13], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[14], 0.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[15], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[16], 1.0 );
    TEST_EQUALITY( mesh_container.coordsBegin()[17], 1.0 );

    // Elements.
    TEST_EQUALITY( *MT::elementsBegin( mesh_container ), 12 );
    TEST_EQUALITY( *mesh_container.elementsBegin(), 12 );

    // Connectivity.
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        TEST_EQUALITY( *( MT::connectivityBegin( mesh_container ) + i ), i );
        TEST_EQUALITY( *( mesh_container.connectivityBegin() + i ), i );
    }

    // Permutation.
    for ( unsigned int i = 0; i < num_vertices; ++i )
    {
        TEST_EQUALITY( *( MT::permutationBegin( mesh_container ) + i ),
                       (int)i );
        TEST_EQUALITY( *( mesh_container.permutationBegin() + i ), (int)i );
    }
}

//---------------------------------------------------------------------------//
// end tstMeshContainer.cpp
//---------------------------------------------------------------------------//
