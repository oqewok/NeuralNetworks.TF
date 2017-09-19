using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ImageConverter
{
	public interface IConverter
	{
		string RootImageFolderName { get; }
		string RootOutputFolderName { get; }
		string FileExtension { get; }
		int ImageCounter { get; set; }

		void Convert(string fullFileName);
	}
}
