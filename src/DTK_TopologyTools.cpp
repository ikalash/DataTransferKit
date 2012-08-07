//---------------------------------------------------------------------------//
/*
  Copyright (c) 2012, Stuart R. Slattery
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  *: Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  *: Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  *: Neither the name of the University of Wisconsin - Madison nor the
  names of its contributors may be used to endorse or promote products
  derived from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
//---------------------------------------------------------------------------//
/*!
 * \file DTK_TopologyTools.cpp
 * \author Stuart R. Slattery
 * \brief TopologyTools definition.
 */
//---------------------------------------------------------------------------//

#include <vector>

#include "DTK_CellTopologyFactory.hpp"
#include "DTK_TopologyTools.hpp"
#include "DTK_Assertion.hpp"

#include <Teuchos_ENull.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_Tuple.hpp>

#include <Shards_CellTopology.hpp>

#include <Intrepid_FieldContainer.hpp>
#include <Intrepid_CellTools.hpp>

namespace DataTransferKit
{
//---------------------------------------------------------------------------//
/*!
 * \brief Point in element query.
 */
bool TopologyTools::pointInElement( Teuchos::Array<double> coords,
				    const moab::EntityHandle element,
				    const Teuchos::RCP<moab::Interface>& moab )
{
    moab::ErrorCode error;

    // Wrap the point in a field container.
    int vertex_dim = coords.size();
    testPrecondition( 0 <= coords.size() && coords.size() <= 3 );

    Teuchos::Tuple<int,2> point_dimensions;
    point_dimensions[0] = 1;
    point_dimensions[1] = vertex_dim;
    Teuchos::ArrayRCP<double> coords_view = Teuchos::arcpFromArray( coords );
    Intrepid::FieldContainer<double> point(
	Teuchos::Array<int>(point_dimensions), coords_view );

    // Get the element topology.
    moab::EntityType element_topology = moab->type_from_handle( element );

    // Get the element vertices.
    std::vector<moab::EntityHandle> element_vertices;
    error = moab->get_adjacencies( &element,
				   1,
				   0,
				   false,
				   element_vertices );
    testInvariant( moab::MB_SUCCESS == error );

    // Extract the vertex coordinates.
    int num_element_vertices = element_vertices.size();
    Teuchos::Array<double> cell_vertex_coords( 3 * num_element_vertices );
    error = moab->get_coords( &element_vertices[0], 
			      element_vertices.size(), 
			      &cell_vertex_coords[0] );
    testInvariant( moab::MB_SUCCESS == error );

    // Typical topology case.
    if ( moab::MBPYRAMID != element_topology )
    {
	// Create the Shards topology for the element type.
	Teuchos::RCP<shards::CellTopology> cell_topo = 
	    CellTopologyFactory::create( element_topology, num_element_vertices );

	// Reduce the dimension of the coordinates if necessary and wrap in a
	// field container. 
	for ( int i = 2; i != vertex_dim-1 ; --i )
	{
	    for ( int n = element_vertices.size() - 1; n > -1; --n )
	    {
		cell_vertex_coords.erase( cell_vertex_coords.begin() + (i+1)*n + i );
	    }
	}

	Teuchos::Tuple<int,3> cell_vertex_dimensions;
	cell_vertex_dimensions[0] = 1;
	cell_vertex_dimensions[1] = num_element_vertices;
	cell_vertex_dimensions[2] = vertex_dim;
	Intrepid::FieldContainer<double> cell_vertices( 
	    Teuchos::Array<int>(cell_vertex_dimensions), 
	    Teuchos::arcpFromArray( cell_vertex_coords ) );

	// Map the point to the reference frame of the cell.
	Intrepid::FieldContainer<double> reference_point( 1, vertex_dim );
	Intrepid::CellTools<double>::mapToReferenceFrame( reference_point,
							  point,
							  cell_vertices,
							  *cell_topo,
							  0 );

	// Check for reference point inclusion in the reference cell.
	return Intrepid::CellTools<double>::checkPointsetInclusion( 
	    reference_point, *cell_topo);
    }

    // We have to handle pyramids differently because Intrepid doesn't
    // currently support them with basis functions. Instead we'll resolve them
    // with two linear tetrahedrons and check for point inclusion in that set
    // instead.
    else
    {
	// Create the Shards topology for the linear tetrahedrons.
	Teuchos::RCP<shards::CellTopology> cell_topo = 
	    CellTopologyFactory::create( moab::MBTET, 4 );

	// Build 2 tetrahedrons from the 1 pyramid.
	testInvariant( vertex_dim == 3 );
	Intrepid::FieldContainer<double> cell_vertices( 1, 4, vertex_dim );

	// Tetrahederon 1.
	cell_vertices( 0, 0, 0 ) = cell_vertex_coords[0];
	cell_vertices( 0, 0, 1 ) = cell_vertex_coords[1];
	cell_vertices( 0, 0, 2 ) = cell_vertex_coords[2];

	cell_vertices( 0, 1, 0 ) = cell_vertex_coords[3];
	cell_vertices( 0, 1, 1 ) = cell_vertex_coords[4];
	cell_vertices( 0, 1, 2 ) = cell_vertex_coords[5];

	cell_vertices( 0, 2, 0 ) = cell_vertex_coords[6];
	cell_vertices( 0, 2, 1 ) = cell_vertex_coords[7];
	cell_vertices( 0, 2, 2 ) = cell_vertex_coords[8];

	cell_vertices( 0, 3, 0 ) = cell_vertex_coords[12];
	cell_vertices( 0, 3, 1 ) = cell_vertex_coords[13];
	cell_vertices( 0, 3, 2 ) = cell_vertex_coords[14];

	// Map the point to the reference frame of the first linear
	// tetrahedron. 
	Intrepid::FieldContainer<double> reference_point( 1, vertex_dim );
	Intrepid::CellTools<double>::mapToReferenceFrame( reference_point,
							  point,
							  cell_vertices,
							  *cell_topo,
							  0 );

	// Check for reference point inclusion in tetrahedron 1.
	bool in_tet_1 = Intrepid::CellTools<double>::checkPointsetInclusion( 
	    reference_point, *cell_topo);

	// If the point is the first tetrahedron, it is in the pyramid and we
	// can exit.
	if ( in_tet_1 )
	{
	    return true;
	}

	// Tetrahederon 2.
	cell_vertices( 0, 0, 0 ) = cell_vertex_coords[0];
	cell_vertices( 0, 0, 1 ) = cell_vertex_coords[1];
	cell_vertices( 0, 0, 2 ) = cell_vertex_coords[2];

	cell_vertices( 0, 1, 0 ) = cell_vertex_coords[6];
	cell_vertices( 0, 1, 1 ) = cell_vertex_coords[7];
	cell_vertices( 0, 1, 2 ) = cell_vertex_coords[8];

	cell_vertices( 0, 2, 0 ) = cell_vertex_coords[9];
	cell_vertices( 0, 2, 1 ) = cell_vertex_coords[10];
	cell_vertices( 0, 2, 2 ) = cell_vertex_coords[11];

	cell_vertices( 0, 3, 0 ) = cell_vertex_coords[12];
	cell_vertices( 0, 3, 1 ) = cell_vertex_coords[13];
	cell_vertices( 0, 3, 2 ) = cell_vertex_coords[14];

	// Map the point to the reference frame of the second linear
	// tetrahedron. 
	Intrepid::CellTools<double>::mapToReferenceFrame( reference_point,
							  point,
							  cell_vertices,
							  *cell_topo,
							  0 );

	// Check for reference point inclusion in tetrahedron 2.
	bool in_tet_2 = Intrepid::CellTools<double>::checkPointsetInclusion( 
	    reference_point, *cell_topo);

	// If the point is the second tetrahedron, it is in the pyramid.
	if ( in_tet_2 )
	{
	    return true;
	}

	return false;
    }
}

//---------------------------------------------------------------------------//

} // end namespace DataTransferKit

//---------------------------------------------------------------------------//
// end DTK_TopologyTools.cpp
//---------------------------------------------------------------------------//