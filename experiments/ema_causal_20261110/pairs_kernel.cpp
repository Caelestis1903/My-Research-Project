#include <algorithm>
#include <cstdint>
#include <vector>
struct Row { double a,b; int y; };
extern "C" void transitions(const double* a,const double* b,const int64_t* y,int64_t n,int64_t* out){
 std::vector<Row> rows(n);std::vector<double> neg;
 for(int64_t i=0;i<n;i++){rows[i]={a[i],b[i],(int)y[i]};if(!y[i])neg.push_back(b[i]);}
 std::sort(neg.begin(),neg.end());std::vector<double> keys=neg;
 keys.erase(std::unique(keys.begin(),keys.end()),keys.end());std::vector<int64_t> tree(keys.size()+1,0);
 auto sum=[&](int64_t k){int64_t s=0;for(;k;k-=k&-k)s+=tree[k];return s;};
 std::sort(rows.begin(),rows.end(),[](const Row& x,const Row& z){return x.a<z.a;});
 int64_t inserted=0;
 for(int64_t i=0;i<n;){int64_t j=i;while(j<n&&rows[j].a==rows[i].a)j++;
   std::vector<double> equal;for(int64_t k=i;k<j;k++)if(!rows[k].y)equal.push_back(rows[k].b);
   std::sort(equal.begin(),equal.end());
   for(int64_t k=i;k<j;k++)if(rows[k].y){double q=rows[k].b;
     auto lo=std::lower_bound(keys.begin(),keys.end(),q)-keys.begin();
     auto hi=std::upper_bound(keys.begin(),keys.end(),q)-keys.begin();
     int64_t less=sum(lo),le=sum(hi),L[3]={inserted-le,le-less,less};
     int64_t el=std::lower_bound(equal.begin(),equal.end(),q)-equal.begin();
     int64_t eh=std::upper_bound(equal.begin(),equal.end(),q)-equal.begin();
     int64_t E[3]={(int64_t)equal.size()-eh,eh-el,el};
     int64_t al=std::lower_bound(neg.begin(),neg.end(),q)-neg.begin();
     int64_t ah=std::upper_bound(neg.begin(),neg.end(),q)-neg.begin();
     int64_t A[3]={(int64_t)neg.size()-ah,ah-al,al};
     for(int z=0;z<3;z++){out[6+z]+=L[z];out[3+z]+=E[z];out[z]+=A[z]-L[z]-E[z];}
   }
   for(double q:equal){int64_t rank=std::lower_bound(keys.begin(),keys.end(),q)-keys.begin()+1;
     for(;rank<(int64_t)tree.size();rank+=rank&-rank)tree[rank]++;inserted++;}
   i=j;
 }
}
