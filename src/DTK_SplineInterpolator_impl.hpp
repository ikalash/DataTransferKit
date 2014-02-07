//---------------------------------------------------------------------------//
/*
  Copyright (c) 2014, Stuart R. Slattery
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  *: Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  *: Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  *: Neither the name of the Oak Ridge National Laboratory nor the
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
 * \file   DTK_SplineInterpolator_impl.hpp
 * \author Stuart R. Slattery
 * \brief  Parallel spline interpolator.
 */
//---------------------------------------------------------------------------//

#ifndef DTK_SPLINEINTERPOLATOR_IMPL_HPP
#define DTK_SPLINEINTERPOLATOR_IMPL_HPP

#include "DTK_DBC.hpp"
#include "DTK_CenterDistributor.hpp"
#include "DTK_SplineInterpolationPairing.hpp"
#include "DTK_SplineOperatorC.hpp"
#include "DTK_SplineOperatorA.hpp"

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_Ptr.hpp>
#include <Teuchos_ArrayRCP.hpp>

#include <Tpetra_Map.hpp>

#include <BelosPseudoBlockGmresSolMgr.hpp>
#include <BelosTpetraAdapter.hpp>
#include <BelosLinearProblem.hpp>

namespace DataTransferKit
{
//---------------------------------------------------------------------------//
/*!
 * \brief Single parameter constructor.
 */
//---------------------------------------------------------------------------//
template<class Basis, class GO, int DIM>
SplineInterpolator<Basis,GO,DIM>::SplineInterpolator( 
    const Teuchos::RCP<const Teuchos::Comm<int> >& comm,
    const Teuchos::ArrayView<const double>& source_centers,
    const Teuchos::ArrayView<const double>& target_centers,
    const double radius,
    const double alpha,
    const Teuchos::RCP<Teuchos::ParameterList>& params )
    : d_comm( comm )
    , d_params( params )
{
    DTK_REQUIRE( 0 == source_centers.size() % DIM );
    DTK_REQUIRE( 0 == target_centers.size() % DIM );

    // Add some additional parameters.
    int verbosityLevel = Belos::IterationDetails | 
			 Belos::OrthoDetails |
			 Belos::FinalSummary |
			 Belos::TimingDetails |
			 Belos::StatusTestDetails |
			 Belos::Warnings | 
			 Belos::Errors;
    d_params->set("Verbosity", verbosityLevel);

    // Build the interpolation and transformation operators.
    buildOperators( source_centers, target_centers, radius, alpha );

    DTK_ENSURE( Teuchos::nonnull(d_C) );
    DTK_ENSURE( Teuchos::nonnull(d_A) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Given a set of scalar values at the given source centers in the
 *  source decomposition, interpolate them onto the target centers in the
 *  target decomposition.
 */
template<class Basis, class GO, int DIM>
void SplineInterpolator<Basis,GO,DIM>::interpolate( 
    const Teuchos::ArrayView<const double>& source_data,
    const int num_source_dims,
    const int source_lda,
    const Teuchos::ArrayView<double>& target_data,
    const int num_target_dims,
    const int target_lda ) const
{
    DTK_REQUIRE( num_source_dims == num_target_dims );
    DTK_REQUIRE( source_data.size() == source_lda * num_source_dims );
    DTK_REQUIRE( target_data.size() == target_lda * num_target_dims );

    // Allocate a work vector.
    MV work_vec( d_C->getDomainMap(), num_source_dims );
    {
	// Copy the source data into a multivector.
	MV source_vec( d_C->getDomainMap(), num_source_dims );
	for ( int i = 0; i < num_source_dims; ++i )
	{
	    Teuchos::ArrayRCP<double> vec_data = source_vec.getDataNonConst(i);

	    if ( 0 == d_comm->getRank() )
	    {
		std::copy( source_data.begin(), source_data.end(),
			   vec_data.begin() + 1 + DIM );
	    }
	    else
	    {
		std::copy( source_data.begin(), source_data.end(), 
			   vec_data.begin() );
	    }
	}

	// Create a linear problem to apply the inverse of the interpolation
	// operator.
	Belos::LinearProblem<double,MV,OP> problem( 
	    d_C, Teuchos::rcpFromRef(work_vec), Teuchos::rcpFromRef(source_vec) );
	problem.setProblem();

	// Create the solver.
	Belos::PseudoBlockGmresSolMgr<double,MV,OP> solver( 
	    Teuchos::rcpFromRef(problem), d_params );

	// Apply the inverse of the interpolation operator.
	Belos::ReturnType rt = solver.solve();
	DTK_INSIST( Belos::Converged == rt );
    }

    // Create a multivector with a view of the target data.
    Teuchos::ArrayRCP<double> target_data_view =
	Teuchos::arcpFromArrayView(target_data);
    Teuchos::RCP<Tpetra::MultiVector<double,int,GO> >
	target_vec = Tpetra::createMultiVectorFromView( 
	    d_A->getRangeMap(), target_data_view, target_lda, num_target_dims );

    // Apply the transformation operator.
    d_A->apply( work_vec, *target_vec );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Build the interpolation and transformation operator.
 */
template<class Basis, class GO, int DIM>
void SplineInterpolator<Basis,GO,DIM>::buildOperators(
    const Teuchos::ArrayView<const double>& source_centers,
    const Teuchos::ArrayView<const double>& target_centers,
    const double radius,
    const double alpha )
{
    // INTERPOLATION OPERATOR.
    // Gather the source centers that are within a radius of the source
    // centers on this  proc.
    Teuchos::Array<double> dist_sources;
    CenterDistributor<DIM> source_distributor( 
	d_comm, source_centers, source_centers, radius, dist_sources );

    // Build the source/source pairings.
    SplineInterpolationPairing<DIM> source_pairings( 
	dist_sources, source_centers, radius );

    // Build the interpolation operator map.
    GO local_num_src = source_centers.size() / DIM;
    GO offset = 0;
    if ( 0 == d_comm->getRank() )
    {
	offset = DIM + 1;
    }
    local_num_src += offset;
    GO global_num_src = 0;
    Teuchos::reduceAll<int,GO>( *d_comm, 
				Teuchos::REDUCE_SUM,
				local_num_src,
				Teuchos::Ptr<GO>(&global_num_src) );

    Teuchos::RCP<const Tpetra::Map<int,GO> > source_map =
	Tpetra::createContigMap<int,GO>( global_num_src,
					 local_num_src,
					 d_comm );

    // Create the source global ids.
    Teuchos::Array<GO> source_gids( local_num_src - offset );
    for ( GO j = 0; j < local_num_src-offset; ++j )
    {
	source_gids[j] = source_map->getMinGlobalIndex() + offset + j;
    }

    // Distribute the global source ids.
    Teuchos::Array<GO> dist_source_gids( 
	source_distributor.getNumImports() );
    Teuchos::ArrayView<const GO> source_gids_view = source_gids();
    Teuchos::ArrayView<GO> dist_gids_view = dist_source_gids();
    source_distributor.distribute( source_gids_view, dist_gids_view );

    // Build the basis.
    Teuchos::RCP<Basis> basis = BP::create( radius );

    // Build the interpolation operator.
    d_C = Teuchos::rcp( 
	new SplineOperatorC<Basis,GO,DIM>( 
	    source_map,
	    source_centers, source_gids,
	    dist_sources, dist_source_gids,
	    Teuchos::rcpFromRef(source_pairings), *basis, alpha) );

    // Cleanup.
    dist_source_gids.clear();
    
    // TRANSFORMATION OPERATOR.
    // Gather the source centers that are within a radius of the target
    // centers on this
    // proc.
    CenterDistributor<DIM> target_distributor( 
	d_comm, source_centers, target_centers, radius, dist_sources  );

    // Distribute the global source ids.
    dist_source_gids.resize( 
	target_distributor.getNumImports() );
    source_gids_view = source_gids();
    dist_gids_view = dist_source_gids();
    target_distributor.distribute( source_gids_view, dist_gids_view );

    // Build the source/target pairings.
    SplineInterpolationPairing<DIM> target_pairings( 
	dist_sources, target_centers, radius );

    // Build the operator map.
    GO local_num_tgt = target_centers.size() / DIM;
    GO global_num_tgt = 0;
    Teuchos::reduceAll<int,GO>( *d_comm, 
				Teuchos::REDUCE_SUM,
				local_num_tgt,
				Teuchos::Ptr<GO>(&global_num_tgt) );

    Teuchos::RCP<const Tpetra::Map<int,GO> > target_map =
	Tpetra::createContigMap<int,GO>( global_num_tgt,
					 local_num_tgt,
					 d_comm );

    // Create the target global ids.
    Teuchos::Array<GO> target_gids( local_num_tgt );
    for ( GO j = 0; j < local_num_tgt; ++j )
    {
	target_gids[j] = target_map->getMinGlobalIndex() + j;
    }

    // Build the transformation operator.
    d_A = SplineOperatorA<Basis,GO,DIM>::create( 
	target_map, source_map,
	target_centers, target_gids,
	dist_sources, dist_source_gids,
	Teuchos::rcpFromRef(target_pairings), *basis, alpha );
}

//---------------------------------------------------------------------------//

} // end namespace DataTransferKit

//---------------------------------------------------------------------------//

#endif // end DTK_SPLINEINTERPOLATOR_IMPL_HPP

//---------------------------------------------------------------------------//
// end DTK_SplineInterpolator_impl.hpp
//---------------------------------------------------------------------------//
