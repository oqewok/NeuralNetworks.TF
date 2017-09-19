using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using OpenCvSharp.CPlusPlus;

namespace ImageConverter
{
	public class XmlFileConverter : IConverter
	{
		private HashSet<string> _filenameDict = new HashSet<string>();
		private List<string> _badFiles = new List<string>(File.ReadAllLines(@"E:\Study\Mallenom\fakes.txt", Encoding.Default));

		public string RootImageFolderName { get; } = @"C:\Users\Павел\YandexDisk\!Выборка для тестирования алгоритмов\";
		public string RootOutputFolderName { get; } = "out";

		public readonly string TrainFilePath = Path.Combine(Directory.GetCurrentDirectory(), "out", "train.txt");

		public XmlFileConverter(string fileExtension)
		{
			FileExtension = fileExtension;
			ImageCounter = 0;
		}

		public string FileExtension { get; } = ".jpeg";
		public int ImageCounter { get; set; }

		public void Convert(string fullFileName)
		{
			var fileInfo = new FileInfo(fullFileName);

			if(_filenameDict.Contains(fileInfo.Name)) return;
			else _filenameDict.Add(fileInfo.Name);

			//if(fileInfo.Extension != FileExtension) return;


			//var opencvMat = mat.Resize(
			//	new OpenCvSharp.CPlusPlus.Size(128, 96),
			//	fx: 1.0,
			//	fy: 1.0,
			//	interpolation: Interpolation.Cubic);

			var outputDir = @"E:\data\XmlMasksNormalized";

			if(!Directory.Exists(outputDir))
			{
				Directory.CreateDirectory(outputDir);
			}

			var sb = new StringBuilder();
			var boundBoxesFileName = Path.Combine(fileInfo.Directory.FullName, fileInfo.Name.Replace(fileInfo.Extension, ".xml"));

			if(!File.Exists(boundBoxesFileName))
			{
				return;
			}

			var outputPath = Path.Combine(outputDir, $"{fileInfo.Name}.txt");

			var rects = new XmlMarkupExporter(boundBoxesFileName)
				.GetPlateRectangles();
				
			if(rects == null)
			{
				return;
			}

			var rect = rects.FirstOrDefault();

			var maskCreator = new ImageMaskCreator(16, 16);

			var mat = new Mat(fullFileName, LoadMode.AnyColor);

			var mask = maskCreator.GetPointCoord(rect, mat.Width, mat.Height);

			if(!_badFiles.Contains(fullFileName))
			{
				mat.Rectangle(new Rect(rect.X, rect.Y, rect.Width, rect.Height), CvColor.GreenYellow, thickness: 2);
				using(var window = new Window(fullFileName, WindowMode.StretchImage, mat))
				{
					window.Move(0, 0);
					window.Resize(1366, 768);
					Cv.WaitKey();
				}
			}

			mat.Dispose();
			//var r = maskCreator.GetPlateRectangles(mask);
			//Рисование изображения
			//var img = maskCreator.DrawMask(mat);
			//img.Rectangle(new Rect(int.Parse(lines[0]), int.Parse(lines[1]), int.Parse(lines[2]) - int.Parse(lines[0]), int.Parse(lines[3]) - int.Parse(lines[1])), CvColor.Red);
			//using(var w = new Window(img))
			//{
			//	Cv.WaitKey();
			//}

			var maskStr = maskCreator.ToString();

			// Запись в файл
			//if(!File.Exists(outputPath))
			//{
			//	AppendText(outputPath, maskStr);
			//}
			//AppendText(TrainFilePath, $@"{fullFileName}  {outputPath}");

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
