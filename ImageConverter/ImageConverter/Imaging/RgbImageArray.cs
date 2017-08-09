using System;
using System.Drawing;
using System.Globalization;
using System.Text;
using OpenCvSharp;
using OpenCvSharp.CPlusPlus;

namespace ImageConverter.Imaging
{
	public class RgbImageArray : IImageArray
	{
		public RgbImageArray(Mat mat)
		{
			Cv2.CvtColor(mat, mat, ColorConversion.BgrToRgb);

			Channels = mat.Channels();
			Width = mat.Rows;
			Height = mat.Cols;

			ChannelsArr = new byte[Channels][][];

			for(int i = 0; i < Channels; i++)
			{
				ChannelsArr[i] = new byte[Width][];

				for(int x = 0; x < Width; x++)
				{
					ChannelsArr[i][x] = new byte[Height];
					for(int y = 0; y < Height; y++)
					{
						var pixel = mat.At<Vec3b>(x, y);

						ChannelsArr[i][x][y] = pixel[i];
					}
				}
			}
		}

		/// <summary> Количество каналов. </summary>
		public int Channels { get; }

		public int Width { get; }
		public int Height { get; }

		/// <summary> Array of channels pixel matrices. </summary>
		public byte[][][] ChannelsArr { get; }

		/// <summary> Индексатор. </summary>
		/// <param name="channel"> Номер канала, начинается с 0. </param>
		/// <returns> Возвращает матрицу пикселей, соответствущую номеру канала. </returns>
		public byte[][] this[int channel]
		{
			get { return ChannelsArr[channel]; }
			set { ChannelsArr[channel] = value; }
		}

		public byte[] ToByteArray()
		{
			int rows = ChannelsArr[0].Length;
			int cols = ChannelsArr[0][0].Length;
			int size = ChannelsArr.Length * rows * cols;
			int index = 0;

			var array = new byte[size];

			for(int i = 0; i < ChannelsArr.Length; i++)
			{
				var matrix = ChannelsArr[i];

				for(int j = 0; j < rows; j++)
				{
					var row = matrix[j];

					Array.ConstrainedCopy(row, 0, array, index, row.Length);

					index += row.Length;
				}				
			}

			return array;
		}

		public double Normalize(int value)
		{
			return value / 255.0;
		}

		public override string ToString()
		{
			var sb = new StringBuilder();

			for(int i = 0; i < Channels; i++)
			{
				for(int x = 0; x < Width; x++)
				{
					for(int y = 0; y < Height; y++)
					{
						sb.Append(Normalize(ChannelsArr[i][x][y]).ToString(CultureInfo.GetCultureInfo("en-US")) + ",");
					}
				}
			}

			return sb.ToString().Substring(0, sb.Length - 1);
		}
	}
}
