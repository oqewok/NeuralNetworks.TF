using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace ImageConverter
{
	public class TxtFileConverter : IConverter
	{
		private HashSet<string> _filenameDict = new HashSet<string>();

		public string RootImageFolderName { get; } = @"E:\data\CropNumbers\Positive";
		public string RootOutputFolderName { get; } = "out";

		public readonly string TrainFilePath = Path.Combine(Directory.GetCurrentDirectory(), "out", "train_classificator.txt");

		public TxtFileConverter(string fileExtension)
		{
			FileExtension = fileExtension;
			ImageCounter = 0;
		}

		public string FileExtension { get; } = ".jpg";
		public int ImageCounter { get; set; }

		public void Convert(string fullFileName)
		{
			var fileInfo = new FileInfo(fullFileName);

			if(!(fileInfo.Extension == FileExtension || fileInfo.Extension == ".jpg" || fileInfo.Extension == ".bmp")) return;

			if(_filenameDict.Contains(fileInfo.Name)) return;
			else _filenameDict.Add(fileInfo.Name);

			//var boundBoxesFileName = Path.Combine(fileInfo.Directory.FullName, fileInfo.Name.Replace(fileInfo.Extension, ".xml"));

			//if(!File.Exists(boundBoxesFileName)) return;

			var outputDir = @"E:\data\pos_labels";

			if(!Directory.Exists(outputDir))
			{
				Directory.CreateDirectory(outputDir);
			}
			
			var sb = new StringBuilder();

			var labelsFileName = fileInfo.Name.Replace(fileInfo.Extension, ".txt");
			var outputPath = Path.Combine(outputDir, labelsFileName);

			var maskCreator = new ImageMaskCreator(16, 16);
			var labels = maskCreator.GetLabelVector(0, 2);
			var maskStr = maskCreator.ToString();

			// Запись в файл
			if(!File.Exists(outputPath))
			{
				AppendText(outputPath, maskStr);
			}

			AppendText(TrainFilePath, $@"{fullFileName}  {outputPath}");

			ImageCounter++;
		}

		public void AppendText(string filename, string text)
		{
			using(var writer = new StreamWriter(filename, true, Encoding.Default))
			{
				writer.WriteLine(text);
				writer.Flush();
			}
		}
	}
}
