#include "checks.h"


void check_valid_tour(const dtype* x_original, const dtype* y_original,const int nc,
	const dtype* x, const dtype* y)
{
	int* checklist = new int[nc];
	for(int i=0;i<nc;++i) checklist[i] = 0;
	for(int i=0;i<nc;++i)
	{
		for(int j=0;j<nc;++j)
		{
			if(x[i] == x_original[j] and y[i] == y_original[j])
			{
				++checklist[j];
				if(checklist[j] > 2)
				{
					fprintf(stdout,"Error! A city (%d) was repeated! Exiting.\n", j);
					delete[] checklist;
					return;
				}
			}
		}
	}
	for(int i =0; i<nc;++i)
	{
		if(checklist[i] != 1)
		{
			fprintf(stdout, "Error! City %d appeared %d time(s).\n",i,checklist[i]);
			delete[] checklist;
			return;
		}
	}
	delete[] checklist;
}

int compute_tour_length(const dtype*x, const dtype*y, const int nc)
{
	int tour_length = 0;
	for(int i=0;i<nc-1;++i)
	{
		dtype dx = x[i+1] - x[i];
		dtype dy = y[i+1] - y[i];
		tour_length += round(sqrt(dx*dx + dy*dy)); 
	}
	dtype dx = x[0] - x[nc-1];
	dtype dy = y[0] - y[nc-1];
	tour_length += round(sqrt(dx*dx + dy*dy)); 
	return tour_length;
}



void print_quadtree(Node* current_node,std::string tabs)
{
	if(current_node->getLevel() == 0)
	{
		fprintf(stdout,"Root Node (Level 0): \n");
	}
	tabs+="\t";
	fprintf(stdout,"%sP: %d\n", tabs.c_str(), current_node->getP());
	fprintf(stdout,"%sS: %d\n", tabs.c_str(), current_node->getS());
	fprintf(stdout,"%sTotal S: %d\n", tabs.c_str(), current_node->getTotalS());
	dtype* com = current_node->getCenterOfMass();
	fprintf( stdout, "%sCenter Of Mass: %f, %f\n", tabs.c_str(), com[0], com[1] );
	// fprintf(stdout,"%sNull Parent: %d\n", tabs.c_str(), current_node->getParent() == NULL);

	leaf_container* pts = current_node->getPoints();
	if(pts->size() > 0)
	{
		fprintf(stdout,"%sPoints List (total: %d): ", tabs.c_str(), (int) pts->size());
		for(size_t i=0;i<pts->size();++i)
		{
			fprintf(stdout,"%d   ", pts->at(i));
		}
		fprintf(stdout,"\n");
	}

	leaf_container* segs = current_node->getSegments();
	if(segs->size() > 0)
	{
		fprintf(stdout,"%sSegments List (total: %d): ", tabs.c_str(), (int) segs->size());
		for(size_t i=0;i<segs->size();++i)
		{
			fprintf(stdout,"%d   ", segs->at(i));
		}
		fprintf(stdout,"\n");
	}
	
	for(int i=0;i<4;++i)
	{
		if(current_node->getChild(i) != NULL)
		{
			fprintf(stdout,"%sQuadrant %d (Level %d):\n",tabs.c_str(),i,current_node->getChild(i)->getLevel());
			print_quadtree(current_node->getChild(i),tabs);
		}
	}
}

void write_tour( const dtype* x, const dtype* y, const int n )
{
	FILE* of = fopen("output.txt","w");
	
	fprintf(of,"%d\n",n);
	for(int i=0;i<n;++i) fprintf(of, "%f %f\n",x[i],y[i]);

	fclose(of);
}