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
 * \file DTK_Rendezvous.hpp
 * \author Stuart R. Slattery
 * \brief Rendezous declaration.
 */
//---------------------------------------------------------------------------//

#ifndef DTK_RENDEZVOUS_HPP
#define DTK_RENDEZVOUS_HPP

#include <set>

#include "DTK_RendezvousMesh.hpp"

#include "DTK_KDTree.hpp"
#include "DTK_RCB.hpp"
#include "DTK_BoundingBox.hpp"
#include "DTK_MeshTraits.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayRCP.hpp>

#include <Tpetra_Map.hpp>

namespace DataTransferKit
{

template<class Mesh>
class Rendezvous
{
    
  public:

    //@{
    //! Typedefs.
    typedef Mesh                                 mesh_type;
    typedef MeshTraits<Mesh>                     MT;
    typedef typename MT::global_ordinal_type     GlobalOrdinal;
    typedef RendezvousMesh<GlobalOrdinal>        RendezvousMeshType;
    typedef Teuchos::RCP<RendezvousMeshType>     RCP_RendezvousMesh;
    typedef KDTree<GlobalOrdinal>                KDTreeType;
    typedef Teuchos::RCP<KDTreeType>             RCP_KDTree;
    typedef RCB<Mesh>                            RCBType;
    typedef Teuchos::RCP<RCBType>                RCP_RCB;
    typedef Teuchos::Comm<int>                   CommType;
    typedef Teuchos::RCP<const CommType>         RCP_Comm;
    typedef Tpetra::Map<GlobalOrdinal>           TpetraMap;
    typedef Teuchos::RCP<const TpetraMap>        RCP_TpetraMap;
    //@}

    // Constructor.
    Rendezvous( const RCP_Comm& comm, const BoundingBox& global_box );

    // Destructor.
    ~Rendezvous();

    // Build the rendezvous decomposition.
    void build( const Mesh& mesh );

    // Get the rendezvous processes for a list of node coordinates.
    Teuchos::Array<int> 
    getRendezvousProcs( const Teuchos::ArrayRCP<double> &coords ) const;

    // Get the native mesh elements in the rendezvous decomposition containing
    // a blocked list of coordinates.
    Teuchos::Array<GlobalOrdinal>
    getElements( const Teuchos::ArrayRCP<double>& coords ) const;

    // Get the rendezvous mesh.
    const RCP_RendezvousMesh& getMesh() const
    { return d_rendezvous_mesh; }

  private:

    // Extract the mesh nodes and elements that are in a bounding box.
    void getMeshInBox( const Mesh& mesh,
		       const BoundingBox& box,
		       Teuchos::Array<int>& nodes_in_box,
		       Teuchos::Array<int>& elements_in_box );

    // Send the mesh to the rendezvous decomposition and build the concrete
    // mesh. 
    void sendMeshToRendezvous( const Mesh& mesh,
			       const Teuchos::Array<int>& elements_in_box );

    // Setup the import communication patterns.
    void setupImportCommunication( 
	const Mesh& mesh,
	const Teuchos::Array<int>& elements_in_box,
	Teuchos::Array<GlobalOrdinal>& rendezvous_nodes,
	Teuchos::Array<GlobalOrdinal>& rendezvous_elements );

  private:

    // Global communicator over which to perform the rendezvous.
    RCP_Comm d_comm;

    // Bounding box in which to perform the rendezvous.
    BoundingBox d_global_box;

    // Mesh node dimension.
    std::size_t d_node_dim;

    // Rendezvous partitioning.
    RCP_RCB d_rcb;

    // Rendezvous on-process mesh.
    RCP_RendezvousMesh d_rendezvous_mesh;

    // Rendezvous on-process kD-tree.
    RCP_KDTree d_kdtree;
};

} // end namespace DataTransferKit

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "DTK_Rendezvous_def.hpp"

#endif // end DTK_RENDEZVOUS_HPP

//---------------------------------------------------------------------------//
// end DTK_Rendezvous.hpp
//---------------------------------------------------------------------------//

