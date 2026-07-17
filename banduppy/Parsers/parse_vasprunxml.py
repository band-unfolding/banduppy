"""
Created on Fri Jul 10 09:14:59 2026

@author: badal.mondal
"""

import numpy as np
import xml.etree.ElementTree as ET
from ..BasicFunctions.general_functions import _BasicFunctionsModule

### =========================================================================== 

class VasprunXml:
    def __init__(self, xml_file):
        self.fname = xml_file
        self.vasprun_data = {}
        _, vasprunfile_too_large = _BasicFunctionsModule._check_file_size(xml_file, _file_size_cutoff=5.0, 
                                                                          _file_size_unit='MB') # 5MB is safe enough
        if vasprunfile_too_large: 
            self._iterparse()
        else:
            self._parse()
        
    def _parse(self) -> None:
        root = ET.parse(self.fname).getroot()
        # Fermi energy
        efermi_elem = root.find(".//dos/i[@name='efermi']")
        if efermi_elem is not None:
            self.vasprun_data['efermi'] = float(efermi_elem.text)
        
        # k-point weights
        kpts_wts = []
        kpt_set = root.find(".//kpoints/varray[@name='weights']")
        if kpt_set is not None:
            for v in kpt_set.findall('v'):
                kpts_wts.append(float(v.text))
            self.vasprun_data['kpoints_wts'] = np.array(kpts_wts)
            
        # dos
        total_dos = []
        total_dos_set = root.find(".//dos/total/array/set/set")
        if total_dos_set is not None:
            for r in total_dos_set.findall('r'):
                # [Energy, total_dos, integrated total dos]
                total_dos.append([float(x) for x in r.text.split()])
            # Scale energy by efermi
            total_dos_array = np.array(total_dos)
            if self.vasprun_data['efermi'] is not None:
                total_dos_array[:,0] -= self.vasprun_data['efermi']
            
            self.vasprun_data['total_dos'] = total_dos_array
            
        return 
            
    def _iterparse(self) -> None:
        efermi, kpts_wts, total_dos = None, [], []
        read_kpt_wt, read_total_dos = False, False
        
        for event, elem in ET.iterparse(self.fname, events=("start", "end")):
            # Fermi energy
            if event == "end" and elem.tag == "i" and elem.attrib.get("name") == 'efermi':
                efermi = float(elem.text)
            
            # k-point weights
            if event == "start" and elem.tag == "varray" and elem.attrib.get("name") == "weights":
                read_kpt_wt = True
            elif event == "end" and elem.tag == "varray" and elem.attrib.get("name") == "weights":
                read_kpt_wt = False
            elif read_kpt_wt and event == "end" and elem.tag =="v":
                kpts_wts.append(float(elem.text))
                
            # k-point weights
            if event == "start" and elem.tag == "total":
                read_total_dos = True
            elif event == "end" and elem.tag == "total":
                read_total_dos = False
            elif read_total_dos and event == "end" and elem.tag =="r":
                total_dos.append([float(x) for x in elem.text.split()])
                
            if event == "end":
                elem.clear()
            
        self.vasprun_data['efermi'] = efermi
        self.vasprun_data['kpoints_wts'] = np.array(kpts_wts)
        # Scale energy by efermi
        total_dos_array = np.array(total_dos)
        if self.vasprun_data['efermi'] is not None:
            total_dos_array[:,0] -= self.vasprun_data['efermi']
        self.vasprun_data['total_dos'] = total_dos_array
        
        return
            
    def _parse_vasprunxml_file_using_ase(self, fname) -> None:
        from ase.io import read
        atms = read(fname, index='-1', format='vasp-xml')
        calc = atms.calc
        self.vasprun_data['efermi'] = calc.get_fermi_level()
        self.vasprun_data['kpoints_wts'] = calc.get_k_point_weights()
        return 