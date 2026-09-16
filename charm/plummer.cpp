/* adapted from barnes benchmark in SPLASH2 */

/*
 * TESTDATA: generate Plummer model initial conditions for test runs,
 * scaled to units such that M = -4E = G = 1 (Henon, Hegge, etc).
 * See Aarseth, SJ, Henon, M, & Wielen, R (1974) Astr & Ap, 37, 183.
 */

#include "Vector3D.h"
#include "Particle.h"
#include <string>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

using namespace std;

#define MFRAC  0.999                /* mass cut off at MFRAC of total */

Real xrand(Real,Real);
void pickshell(Vector3D<Real> &vec, Real rad);

Particle *testdata(int nbody)
{
   Real rsc, vsc, rsq, r, v, x, y;
   Vector3D<Real> cmr, cmv;
   Particle *p;
   int rejects = 0;
   int k;
   int halfnbody, i;
   Real offset;
   Particle *cp;
   Real tmp;

   Particle *bodytab = new Particle[nbody * sizeof(Particle)];
   assert(bodytab != NULL);

   rsc = 9 * PI / 16;
   vsc = sqrt(1.0 / rsc);

   cmr = Vector3D<Real>(0.0);
   cmv = Vector3D<Real>(0.0);

   halfnbody = nbody / 2;
   if (nbody % 2 != 0) halfnbody++;
   for (p = bodytab; p < bodytab+halfnbody; p++) {
      p->mass = 1.0/nbody;
      r = 1 / sqrt(std::pow((double)xrand(0.0, MFRAC), (double)-2.0/3.0) - 1);
      /*   reject radii greater than 10 */
      while (r > 9.0) {
	 rejects++;
	 r = 1 / sqrt(std::pow((double)xrand(0.0, MFRAC), (double)-2.0/3.0) - 1);

      }        
      pickshell(p->position, rsc * r);
      
      cmr += p->position;
      do {
	 x = xrand(0.0, 1.0);
	 y = xrand(0.0, 0.1);

      } while (y > x*x * std::pow((double)(1 - x*x), (double)3.5));

      v = sqrt(2.0) * x / std::pow((double)(1 + r*r), (double)0.25);
      pickshell(p->velocity, vsc * v);
      cmv += p->velocity;
      
   }

   offset = 4.0;

   for (p = bodytab + halfnbody; p < bodytab+nbody; p++) {
      p->mass = 1.0 / nbody;
      cp = p - halfnbody;
      p->position = cp->position+offset;
      p->velocity = cp->velocity;
      cmr += p->position;
      /* NOTE: this read `cmv += p->position`. Accumulating positions into the
       * velocity centre of mass gave every particle a spurious bulk drift of
       * roughly offset/2 per axis once cmv was subtracted below, which by
       * itself accounted for ~6 of the ~6.1 total kinetic energy in a 10k-body
       * run. */
      cmv += p->velocity;
   }

   cmr /= nbody;
   cmv /= nbody;

   for (p = bodytab; p < bodytab+nbody; p++) {
     p->position -= cmr;
     p->velocity -= cmv;
   }

   return bodytab;
}

/*
 * PICKSHELL: pick a random point on a sphere of specified radius.
 */

void pickshell(Vector3D<Real> &vec, Real rad)
{
   register int k;
   Real rsq, rsc;

   do {
     vec.x = xrand(-1.0,1.0);
     vec.y = xrand(-1.0,1.0);
     vec.z = xrand(-1.0,1.0);
     rsq = vec.lengthSquared();
   } while (rsq > 1.0);

   rsc = rad / sqrt(rsq);
   vec = rsc*vec;
}


int intpow(int i, int j)
{   
    int k;
    int temp = 1;

    for (k = 0; k < j; k++)
        temp = temp*i;
    return temp;
}

void pranset(int);

/* Writes the binary layout the simulation reads by default: two ints
 * (nbody, ndims), one Real (tnow), then REALS_PER_PARTICLE Reals per body.  */
static void writeBinary(const char *fname, Particle *p, int nbody){
  int ndims = 3;
  Real tnow = 0.0;
  ofstream out(fname, ios::out|ios::binary);

  out.write((char *)&nbody, sizeof(int));
  out.write((char *)&ndims, sizeof(int));
  out.write((char *)&tnow, sizeof(Real));

  Real tmp[REALS_PER_PARTICLE];
  Real soft = 0.001;

  for(int i = 0; i < nbody; i++, p++){
    tmp[0] = p->position.x;
    tmp[1] = p->position.y;
    tmp[2] = p->position.z;
    tmp[3] = p->velocity.x;
    tmp[4] = p->velocity.y;
    tmp[5] = p->velocity.z;
    tmp[6] = p->mass;
    tmp[7] = soft;

    out.write((char*)tmp, REALS_PER_PARTICLE*sizeof(Real));
  }

  out.close();
}

/* Writes the benchmark's portable format (spec v1.0 sec. 5), matching the
 * columns https://github.com/vancraar/DataGenerator emits. Enough digits to
 * round-trip a double, so the two formats give the same answers.  */
static void writeCsv(const char *fname, Particle *p, int nbody){
  ofstream out(fname);
  out.precision(17);

  out << "id,mass,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z\n";
  for(int i = 0; i < nbody; i++, p++){
    out << i << ','
        << p->mass << ','
        << p->position.x << ',' << p->position.y << ',' << p->position.z << ','
        << p->velocity.x << ',' << p->velocity.y << ',' << p->velocity.z << '\n';
  }

  out.close();
}

int main(int argc, char **argv){
  pranset(128363);
  if(argc < 3 || argc > 4){
    cerr << "usage: " << argv[0] << " <nbody> <outfile> [binary|csv]" << endl;
    cerr << "       format defaults to binary" << endl;
    return 1;
  }

  int nbody = atoi(argv[1]);
  string format = (argc == 4) ? string(argv[3]) : string("binary");

  Particle *p = testdata(nbody);

  if(format == "csv"){
    writeCsv(argv[2], p, nbody);
  }
  else if(format == "binary" || format == "bin"){
    writeBinary(argv[2], p, nbody);
  }
  else{
    cerr << "unrecognized format '" << format << "' (expected binary or csv)" << endl;
    return 1;
  }

  cout << "wrote " << nbody << " bodies to " << argv[2]
       << " as " << format << endl;

  return 0;
}


