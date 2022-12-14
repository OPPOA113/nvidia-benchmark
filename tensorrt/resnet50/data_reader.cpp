#include "data_reader.hpp" 
#include <sstream>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <map>

#include <algorithm>

using namespace std;

namespace Tn
{
    list<string> readFileList(const string& fileName)
    {
        ifstream file(fileName);  
        if(!file.is_open())
        {
            cout << "read file list error,please check file :" << fileName << endl;
            exit(-1);
        }

        string strLine;  
        list<string> files;
        while( getline(file,strLine) )                               
            files.push_back(strLine);

        file.close();

        return files;
    }

    list<Source> readLabelFileList(const string& fileName)
    {
        ifstream file(fileName);  
        if(!file.is_open())
        {
            cout << "read file list error,please check file :" << fileName << endl;
            exit(-1);
        }

        string strLine;  
        list<Source> result;
        while(!file.eof())
        {
            Source data;
            file >> data.fileName >> data.label;
            result.emplace_back(data);
        } 

        return result;
    }

    vector<string> split(const string& str, char delim)
    {
        stringstream ss(str);
        string token;
        vector<string> container;
        while (getline(ss, token, delim)) {
            container.push_back(token);
        }

        return container;
    }

    std::tuple<std::list<std::string>, std::list<std::vector<Bbox>>> readObjectLabelFileList(const string& fileName)
    {
        list<string> fileList;
        list<vector<Bbox>> bBoxes;

        ifstream file(fileName);  
        if(!file.is_open())
        {
            cout << "read file list error,please check file :" << fileName << endl;
            exit(-1);
        }

        string strLine;  
        while( getline(file,strLine) )                               
        { 
            vector<string> line=split(strLine, '\n');
            if(line.size() < 1)
                continue;
            vector<string> strs=split(line[0], ' ');            

            int idx = 0;
            string dataName=strs[idx++];  

            int trueBoxCount = (strs.size() - 1)/2;
            vector<Bbox> truthboxes;
            truthboxes.reserve(trueBoxCount);
            for (int i = 0 ;i < trueBoxCount ;++i)
            {
                //class
                string classId = strs[idx++];
                
                //bbox Length
                int length = strs[idx].length();
                //remove bracket [ ]
                string bbox = strs[idx++].substr(1,length-2);

                vector<string> strs_txt = split(bbox, ','); 
                Bbox truthbox;
                truthbox.classId = stoi(classId);
                truthbox.left = stof(strs_txt[0]);
                truthbox.top = stof(strs_txt[1]);
                truthbox.right = truthbox.left + stof(strs_txt[2]);
                truthbox.bot = truthbox.top + stof(strs_txt[3]);

                truthboxes.push_back(truthbox);
            }
            
            fileList.emplace_back(dataName);
            bBoxes.emplace_back(truthboxes);
        } 

        file.close();

        return make_tuple(move(fileList),move(bBoxes));
    }

    void GetRealLabelMap(std::string fileName, std::map<string, string> &val_label_map)
    {
        // std::map<string, string> val_label_map;
        ifstream file(fileName);  
        if(!file.is_open())
        {
            cout << "read file list error,please check file :" << fileName << endl;
            exit(-1);
        }
        string strLine;  
        while( getline(file,strLine) )                               
        { 
            // strLine.erase(std::remove(strLine.begin(), strLine.end(), '\n'), strLine.end());
            vector<string> line=split(strLine, ' ');
            if(line.size() < 2)
                continue;
            val_label_map.insert(pair<string, string>(line[0], line[1]));
        } 
        file.close();
    }
    
}
