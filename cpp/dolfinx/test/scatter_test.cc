#include <iostream>
#include <span>
#include <vector>

#include "dolfinx.h"
#include "dolfinx/io/XDMFFile.h"

// twoscale
#include "util.h"

using namespace std;

int main(int argc, char *argv[])
{
   dolfinx::init_logging(argc, argv);
   {
      // init
      MPI_Init(&argc, &argv);
      const std::size_t nb_proc = dolfinx::MPI::size(MPI_COMM_WORLD);
      const int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
      const bool do_print = (!mpi_rank);

      string no = "proc_" + std::to_string(mpi_rank) + "_output.txt";
      if (mpi_rank > 5)
         auto fd = freopen("/dev/null", "w", stdout);
      else
         auto fd = freopen(no.c_str(), "w", stdout);

      cout << " nb_proc " << nb_proc << " mpi_rank " << mpi_rank << endl;

      // element
      auto tria = dolfinx::fem::CoordinateElement<double>(dolfinx::mesh::CellType::triangle, 1);
      int dim = dolfinx::mesh::cell_dim(dolfinx::mesh::CellType::triangle);
      int dim1 = dolfinx::mesh::cell_dim(dolfinx::mesh::CellType::interval);
      int dim0 = dolfinx::mesh::cell_dim(dolfinx::mesh::CellType::point);

      // read mesh and tag
      dolfinx::io::XDMFFile file(MPI_COMM_WORLD, "data/square.xdmf", "r");
      dolfinx::mesh::Mesh<double> domain = file.read_mesh(tria, dolfinx::mesh::GhostMode::none, "neper_dam");
      dolfinx::mesh::MeshTags<int32_t> cell_tag = file.read_meshtags(domain, "neper_dam_cells", {});
      domain.topology_mutable()->create_entities(dim1);
      domain.topology_mutable()->create_connectivity(dim1, dim);
      dolfinx::mesh::MeshTags<int32_t> edge_tag = file.read_meshtags(domain, "neper_dam_facets", {});

      // face idx
      auto idxmap1 = domain.topology()->index_map(dim1);

      // fill a vector coreponding to faces with adjacent cell tag
      const int32_t nbd = dim;
      const std::int32_t nbfl = idxmap1->size_local();
      const std::int32_t nbf = nbfl + idxmap1->num_ghosts();

      auto adj12 = domain.topology()->connectivity(dim1, dim);

      std::vector<std::int32_t> faceadj(nbf * nbd, -1);

      auto fill = [&cell_tag, &nbf, &nbd, &adj12, &faceadj]() {
         std::fill(faceadj.begin(), faceadj.end(), -1);
         auto ctv = cell_tag.values();
         for (size_t i = 0; i < nbf; ++i)
         {
            auto cells = adj12->links(i);
            for (auto c : cells)
            {
               std::int32_t *p = &faceadj[i * nbd];
               if (*p == -1)
                  *p = ctv[c];
               else
               {
                  ++p;
                  if (*p == -1)
                     *p = ctv[c];
                  else
                     MPI_Abort(MPI_COMM_WORLD, -65);
               }
            }
         }
      };
      fill();
      cout << " nbf " << nbf << endl;
      cout << "before ";
      auto print = [&faceadj, &nbf, &nbfl, &nbd]() {
         for (size_t i = 0; i < nbf; ++i)
         {
            if (i == nbfl) cout << "$$|";
            std::int32_t *p = &faceadj[i * nbd];
            for (size_t j = 0; j < nbd; ++j) cout << " " << std::setw(2) << p[j];
            cout << " |";
         }
         cout << endl;
      };

      print();

      // scatterer
      dolfinx::common::Scatterer scatterer(*idxmap1, nbd);

      // exchange operator for rev
      auto exch = [&faceadj, &nbd](const std::int32_t &i, const std::int32_t &j) {
         // works only because i is comming as a reference from span
         std::int32_t rf = (&i - faceadj.data()) / nbd;
         std::span<std::int32_t> face_tags(faceadj.data() + rf * nbd, nbd);
         if (j > -1)
         {
            // if j not present add it
            auto itf = std::ranges::find(face_tags, j);
            if (itf == face_tags.end())
            {
               auto itfb = std::ranges::find(face_tags, -1);
               assert(itfb != face_tags.end());
               *itfb = j;
            }
            // if j present check no -1 present. If yes add j
            else
            {
               auto itfb = std::ranges::find(face_tags, -1);
               if (itfb != face_tags.end()) *itfb = j;
            }
         }
         return i;
      };
      // exchange face tags: rev followed by fwd
      twoscale_dolfinx::scatter_rev_fwd(scatterer, std::span<std::int32_t>(faceadj.begin(), faceadj.begin() + nbfl * nbd),
                                        std::span<std::int32_t>(faceadj.begin() + nbfl * nbd, faceadj.end()), exch);

      cout << "after  ";
      print();

      // check
      auto check = [&faceadj, &edge_tag, &nbd]() {
         auto etv = edge_tag.values();
         auto eti = edge_tag.indices();
         size_t k = 0;
         for (auto e : eti)
         {
            std::int32_t *p = &faceadj[e * nbd];
            cout << e << " " << etv[k] << " " << p[0] << " " << p[1] << endl;
            // inter cell_tag bnd
            if (etv[k] == 4)
            {
               if (p[0] == p[1])
               {
                  cout << "edge " << e << " is taged as 4 but got same cell tag ?" << endl;
                  MPI_Abort(MPI_COMM_WORLD, -34);
               }
               if (p[0] == -1 || p[1] == -1)
               {
                  cout << "edge " << e << " is taged as 4 but got only one cell tag ?" << endl;
                  MPI_Abort(MPI_COMM_WORLD, -35);
               }
            }
            // part bnd
            else
            {
               if (p[0] == p[1])
               {
                  cout << "edge " << e << " is taged as bnd but got same cell tag ?" << endl;
                  MPI_Abort(MPI_COMM_WORLD, -36);
               }
            }
            ++k;
         }
      };
      check();

      // reset
      fill();

      cout << "reset ";
      print();

      // exchange face tags: first rev
      twoscale_dolfinx::scatter_rev(scatterer, std::span<std::int32_t>(faceadj.begin(), faceadj.begin() + nbfl * nbd),
                                    std::span<const std::int32_t>(faceadj.begin() + nbfl * nbd, faceadj.end()), exch);

      cout << "rev   ";
      print();
      // exchange face tags: fwd
      twoscale_dolfinx::scatter_fwd(scatterer, std::span<const std::int32_t>(faceadj.begin(), faceadj.begin() + nbfl * nbd),
                                    std::span<std::int32_t>(faceadj.begin() + nbfl * nbd, faceadj.end()));

      cout << "fwd   ";
      print();
      check();
   }

   MPI_Finalize();
   return 0;
}
