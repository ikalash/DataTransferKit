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
 * \brief DTK_ClassicMeshNodalShapeFunction_impl.hpp
 * \author Stuart R. Slattery
 * \brief Nodal shape function implementation for Classic mesh.
 */
//---------------------------------------------------------------------------//

#ifndef DTK_CLASSICMESHNODALSHAPEFUNCTION_IMPL_HPP
#define DTK_CLASSICMESHNODALSHAPEFUNCTION_IMPL_HPP

#include "DTK_ClassicMeshElementExtraData.hpp"
#include "DTK_DBC.hpp"
#include "DTK_IntrepidBasisFactory.hpp"

#include <Shards_CellTopology.hpp>

namespace DataTransferKit
{
//---------------------------------------------------------------------------//
// Constructor.
template <class Mesh>
ClassicMeshNodalShapeFunction<Mesh>::ClassicMeshNodalShapeFunction(
    const Teuchos::RCP<ClassicMesh<Mesh>> &mesh )
    : d_mesh( mesh )
{ /* ... */
}

//---------------------------------------------------------------------------//
// Given an entity, get the ids of the degrees of freedom in the vector space
// supporting its shape function.
template <class Mesh>
void ClassicMeshNodalShapeFunction<Mesh>::entitySupportIds(
    const Entity &entity, Teuchos::Array<SupportId> &support_ids ) const
{
    int block_id = Teuchos::rcp_dynamic_cast<ClassicMeshElementExtraData>(
                       entity.extraData() )
                       ->d_block_id;
    support_ids = d_mesh->getElementConnectivity( entity.id(), block_id );
}

//---------------------------------------------------------------------------//
// Given an entity and a reference point, evaluate the shape function of the
// entity at that point.
template <class Mesh>
void ClassicMeshNodalShapeFunction<Mesh>::evaluateValue(
    const Entity &entity,
    const Teuchos::ArrayView<const double> &reference_point,
    Teuchos::Array<double> &values ) const
{
    // Get the basis for the entity.
    Teuchos::RCP<Intrepid2::Basis<Kokkos::HostSpace, double, double>>
        basis = getIntrepidBasis( entity );

    // Wrap the reference point.
    int num_points = 1;
    int space_dim = reference_point.size();
    Kokkos::DynRankView<double, Kokkos::HostSpace> point_container(
        "point_container", num_points, space_dim );
    for ( int d = 0; d < space_dim; ++d )
    {
        point_container( 0, d ) = reference_point[d];
    }

    // Evaluate the basis function.
    int cardinality = basis->getCardinality();
    values.resize( cardinality );
    Kokkos::DynRankView<double, Kokkos::HostSpace> value_container(
        "value_container", cardinality, num_points );
    basis->getValues( value_container, point_container,
                      Intrepid2::OPERATOR_VALUE );

    // Extract the evaluations.
    for ( int n = 0; n < cardinality; ++n )
    {
        values[n] = value_container( n, 0 );
    }
}

//---------------------------------------------------------------------------//
// Given an entity and a reference point, evaluate the gradient of the shape
// function of the entity at that point.
template <class Mesh>
void ClassicMeshNodalShapeFunction<Mesh>::evaluateGradient(
    const Entity &entity,
    const Teuchos::ArrayView<const double> &reference_point,
    Teuchos::Array<Teuchos::Array<double>> &gradients ) const
{
    // Get the basis for the entity.
    Teuchos::RCP<Intrepid2::Basis<Kokkos::HostSpace, double, double>>
        basis = getIntrepidBasis( entity );

    // Wrap the reference point.
    int space_dim = reference_point.size();
    Kokkos::DynRankView<double, Kokkos::HostSpace> point_container(
        "point_container", 1, space_dim );
    for ( int d = 0; d < space_dim; ++d )
    {
        point_container( 0, d ) = reference_point[d];
    }

    // Evaluate the basis function.
    int cardinality = basis->getCardinality();
    Kokkos::DynRankView<double, Kokkos::HostSpace> grad_container(
        "grad_container", cardinality, 1, space_dim );
    basis->getValues( grad_container, point_container,
                      Intrepid2::OPERATOR_GRAD );

    // Extract the evaluations.
    gradients.resize( cardinality );
    for ( int n = 0; n < cardinality; ++n )
    {
        gradients[n].resize( space_dim );
        for ( int d = 0; d < space_dim; ++d )
        {
            gradients[n][d] = grad_container( n, 0, d );
        }
    }
}

//---------------------------------------------------------------------------//
// Given an entity, get the intrepid basis function.
template <class Mesh>
Teuchos::RCP<Intrepid2::Basis<Kokkos::HostSpace, double, double>>
ClassicMeshNodalShapeFunction<Mesh>::getIntrepidBasis(
    const Entity &entity ) const
{
    int block_id = Teuchos::rcp_dynamic_cast<ClassicMeshElementExtraData>(
                       entity.extraData() )
                       ->d_block_id;
    shards::CellTopology entity_topo = d_mesh->getBlockTopology( block_id );
    return IntrepidBasisFactory::create( entity_topo );
}

//---------------------------------------------------------------------------//

} // end namespace DataTransferKit

#endif // end DTK_CLASSICMESHNODALSHAPEFUNCTION_IMPL_HPP

//---------------------------------------------------------------------------//
// end DTK_ClassicMeshNodalShapeFunction_impl.hpp
//---------------------------------------------------------------------------//
