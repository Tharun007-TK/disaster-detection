const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function POST(request: Request) {
  const formData = await request.formData();
  const res = await fetch(`${API_URL}/api/inference/single`, {
    method: "POST",
    body: formData,
  });
  const data = await res.json();
  return Response.json(data, { status: res.status });
}
