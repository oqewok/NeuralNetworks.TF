using System;
using System.Drawing;
using System.IO;
using System.Text;

using ImageConverter.Imaging;

using OpenCvSharp;
using OpenCvSharp.CPlusPlus;

namespace ImageConverter
{
	public class CsvFileConverter
	{
		public const string RootImageFolderName  = "images";
		public const string RootOutputFolderName = "out";
		public const string OutputFileName       = "data.csv";
		
		public CsvFileConverter(string fileExtension)
		{
			ImageCounter = 0;
			FileExtension = fileExtension;
		}

		public string FileExtension { get; }
		public int ImageCounter { get; private set; }

		public void Convert(string fullFileName)
		{
			var fileInfo = new FileInfo(fullFileName);

			if(fileInfo.Extension != FileExtension) return;

			var mat = new Mat(fullFileName, LoadMode.Color);

			var opencvMat = mat.Resize(
				new OpenCvSharp.CPlusPlus.Size(128, 96),
				fx: 1.0,
				fy: 1.0,
				interpolation: Interpolation.Cubic);

			var outputDir = Path.Combine(Directory.GetCurrentDirectory(), RootOutputFolderName);

			if(!Directory.Exists(outputDir))
			{
				Directory.CreateDirectory(outputDir);
			}
			
			var outputPath = Path.Combine(outputDir, OutputFileName);
			var sb = new StringBuilder();
			var boundBoxesDir = @"E:\data\gt_db\labels";
			var boundBoxesFileNames = Directory
					.GetFiles(boundBoxesDir);

			//var list = ImageMaskCreator.GetRegionsRectangles(8, 8);

			var imageArr = new RgbImageArray(opencvMat);
			var lines = File.ReadAllText(boundBoxesFileNames[ImageCounter]).Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

			var maskCreator = new ImageMaskCreator(16, 16);
			var mask = maskCreator.FillMasks(new Rectangle(int.Parse(lines[0]), int.Parse(lines[1]), int.Parse(lines[2]) - int.Parse(lines[0]), int.Parse(lines[3]) - int.Parse(lines[1])));
			//var mask = ImageMaskCreator.CreateStringMask(new Rectangle[] { new Rectangle(int.Parse(lines[0]), int.Parse(lines[1]), int.Parse(lines[2]) - int.Parse(lines[0]), int.Parse(lines[3]) - int.Parse(lines[1])) });

			//Рисование изображения
			var img = maskCreator.DrawMask(mat);
			img.Rectangle(new Rect(int.Parse(lines[0]), int.Parse(lines[1]), int.Parse(lines[2]) - int.Parse(lines[0]), int.Parse(lines[3]) - int.Parse(lines[1])), CvColor.Red);
			using(var w = new Window(img))
			{
				Cv.WaitKey();
			}

			sb.Append(imageArr.ToString());

			var maskStr = maskCreator.ToString();

			//AppendText(Path.Combine(outputDir, "masks.csv"), maskStr);
			//AppendText(outputPath, sb.ToString());

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
