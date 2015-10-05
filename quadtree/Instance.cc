#include "Instance.h"

// This simply fills the cities_ private variable and then quits.
// This also quits if the header or end of file is reached.
void Instance::readCities(string file_name)
{
  string header_end("NODE_COORD_SECTION");
  string cities_label("DIMENSION");
  ifstream file_stream(file_name.c_str());
  while ( !file_stream.eof() )
  {
    string line;
    getline(file_stream, line);
    if ( line.find(header_end) != string::npos ) break;

    if ( line.find(cities_label) != string::npos )
    {
      stringstream line_stream(line);
      // Now we exhaust the string stream to the last entry, which will be the 
      // city count.
      string string_buffer;
      line_stream >> string_buffer;
      line_stream >> string_buffer;
      line_stream >> string_buffer;
      line_stream >> string_buffer;
      cities_ = stoi(string_buffer);
      break;
    }
  }
  cout << "Cities read: " << cities_ << endl;
}

// This fills x_ and y_. The knowledge of cities_ is assumed.
void Instance::readCoordinates(string file_name)
{
  string header_end("NODE_COORD_SECTION");
  ifstream file_stream(file_name.c_str());

  //Let us skip the header.
  while ( !file_stream.eof() )
  {
    string line;
    getline(file_stream, line);
    if ( line.find(header_end) != string::npos ) break;
  }

  // Now let us fill the coordinates.
  int counter = 1;
  while ( !file_stream.eof() )
  {
    string line;
    getline(file_stream, line);
    stringstream line_stream(line);

    int city_id;
    line_stream >> city_id;

    if (city_id == counter)
    {
      line_stream >> x_[city_id-1];
      line_stream >> y_[city_id-1];
      ++counter;
    }
    // cout << "Read city " << city_id << " / " << cities_ << endl;
    if(city_id >= cities_) break; 
  }
  cout << "Done reading city file." << endl;
  file_stream.close();
}