//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   core/Transfer_Map.hh
 * \author Stuart Slattery
 * \date   Thu Oct 06 15:53:09 2011
 * \brief  Transfer_Map class definition.
 */
//---------------------------------------------------------------------------//
// $Id: template.hh,v 1.4 2008/01/02 17:18:47 9te Exp $
//---------------------------------------------------------------------------//

#ifndef coupler_Transfer_Map_hh
#define coupler_Transfer_Map_hh

#include <map>
#include <set>

namespace coupler
{

//===========================================================================//
/*!
 * \class Transfer_Map
 * \brief A basic map data structure to hold topological relationships between
 * parallel meshes.
 *
 * For one way transfer, the source application must know which of its local
 * mesh entities, identified here by a handle, must map onto a mesh entity in
 * the target application and on what parallel process rank of that target
 * application that entity exists. The target application must know the
 * inverse of this; for its local target points, it needs to know which source
 * processes will be sending data to map onto those points, and which pieces
 * of data correspond to those points.
 */
//===========================================================================//

class Transfer_Map 
{
  public:

    //@{
    //! Useful typedefs.
    typedef std::multimap<int,int>                       Map;
    typedef Map::const_iterator                          Map_Iterator;
    typedef std::pair<Map_Iterator, Map_Iterator>        Iterator_Pair;
    typedef std::pair<int,int>                           Map_Pair;
    typedef std::set<int>                                Set;
    typedef Set::const_iterator                          Set_Iterator;
    //@}

  private:

    // Map for the source application. 
    // Key: target rank, Value: source handle. 
    Map source_map;

    // Map for the target application. 
    // Key: source rank, Value: target handle.
    Map target_map;

    // Set of unique source ranks.
    Set source_set;

    // Set of unique target ranks.
    Set target_set;

  public:

    //! Constructor.
    Transfer_Map();

    //! Destructor.
    ~Transfer_Map();

    //! Add a pair to the source map. The source handle corresponding to an
    //! entity in the local domain correlates to the range owned by the
    //! target rank.
    void add_domain_pair(int target_rank, int source_handle);

    //! Add a pair to the target map. The target handle corresponding to an
    //! entity in the local range correlates to the domain owned by the
    //! source rank.
    void add_range_pair(int source_rank, int target_handle);

    //! Get the number of source handles with a specific target rank. This is
    //! the size of the local domain correlating to the target rank.
    int domain_size(int target_rank);
    
    //! Get the number of target handles with a specific source rank. This is
    //! the size of the local range correlating to the source rank.
    int range_size(int source_rank);

    //! Get the iterator pair for the source domain of a target rank. This
    //! correlates to the local source handles that exist in the range of the
    //! target rank.
    Iterator_Pair source_domain(int target_rank);

    //! Get the iterator pair for the target range of a source rank. This
    //! correlates to the local target handles that exist in the domain of the
    //! source rank.
    Iterator_Pair target_range(int source_rank);

    //! Return a const_iterator to the beginning of the source rank set.
    Set_Iterator source_set_begin();

    //! Return a const_iterator to the end of the source rank set.
    Set_Iterator source_set_end();

    //! Return a const_iterator to the beginning of the target rank set.
    Set_Iterator target_set_begin();
    
    //! Return a const_iterator to the end of the target rank set.
    Set_Iterator target_set_end();
};

} // end namespace coupler

#endif // coupler_Transfer_Map_hh

//---------------------------------------------------------------------------//
//              end of core/Transfer_Map.hh
//---------------------------------------------------------------------------//
